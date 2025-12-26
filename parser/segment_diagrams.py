import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Tuple

import fitz  # pymupdf
import cv2
import numpy as np
from PIL import Image

from openai import OpenAI


# -----------------------------
# Configuration
# -----------------------------

MODEL = "gpt-5.2"   # vision-capable model
INK_PREVIEW_WIDTH = 1024
SEG_DPI = 220
EXPORT_DPI = 400


# -----------------------------
# PDF rendering
# -----------------------------

def render_pdf_page(pdf_path: str, page_index: int, dpi: int) -> Image.Image:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()


# -----------------------------
# Ink + border removal
# -----------------------------

def build_ink_no_border(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    ink = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )

    ink = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    # remove border explicitly
    h, w = ink.shape
    margin = 10
    ink[:margin, :] = 0
    ink[-margin:, :] = 0
    ink[:, :margin] = 0
    ink[:, -margin:] = 0

    return ink


# -----------------------------
# VLM call (polygons only)
# -----------------------------

def call_vlm_for_polygons(ink_preview_png: Path) -> dict:
    client = OpenAI()

    import base64, json

    with open(ink_preview_png, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    # Use a data URL (THIS is the key change)
    image_data_url = f"data:image/png;base64,{b64}"

    prompt = """
Return ONLY valid JSON. No markdown. No extra keys. No explanations.

You are given a black-and-white technical drawing page (line art only). The page contains MULTIPLE separate diagrams/views (e.g., top view, side view, section view, isometric view), plus tables/spec blocks/title blocks.

GOAL:
Extract EACH individual diagram/view as its own item. DO NOT combine multiple views into one output item.

DEFINITIONS:
- “One diagram/view” = one logical illustration of the part in a single view or depiction.
  Examples: a single orthographic view, a single section view, a single isometric view.
- If two depictions are separated by whitespace and could stand alone, they MUST be separate diagrams.
- If a depiction includes dimension arrows/leader lines/text that clearly belongs to THAT view, include it.
- If a table/spec/title block is nearby, exclude it.

STRICT RULES (highest priority):
1) NEVER group multiple views/depictions into the same diagram.
2) Prefer OVER-SPLITTING to under-splitting.
   If uncertain whether two regions are the same diagram, output them as two separate diagrams.
3) Each diagram polygon set must cover ONLY one view/depiction and should not overlap into neighboring views.
4) Do NOT use bounding boxes. Use polygons only.
5) Polygons should be generous around the single view (include ~3–8% margin) but MUST NOT include adjacent views.

OUTPUT FORMAT:
- Use normalized coordinates in [0,1].
- Each diagram has an id and one or more polygons.
- Use MULTIPLE polygons only when a single view is disconnected (e.g., thin lines), but they must still belong to the SAME view.
- If there are 5 views, output 5 diagram items.

SCHEMA (must match exactly):
{
  "diagrams": [
    {
      "id": "diagram_1",
      "polygons": [
        [[x,y],[x,y],...]
      ]
    }
  ]
}

EXCLUDE COMPLETELY:
- tables (dimension tables)
- specification text blocks
- title blocks / logos / contact info
- page border/frame
- watermarks

FINAL CHECK BEFORE YOU RESPOND:
- Each “diagram_i” contains ONLY ONE view/depiction.
- If any diagram includes parts of two views, split it into two diagrams.
- Prefer more diagrams rather than fewer.

"""

    resp = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
        max_output_tokens=2500,
    )

    raw = resp.output_text
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON. Model said:\n{raw}") from e




# -----------------------------
# Polygon → mask
# -----------------------------

def polygons_to_mask(
    h: int,
    w: int,
    polygons_norm: List[List[List[float]]]
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)

    for poly in polygons_norm:
        pts = np.array(
            [[int(x * w), int(y * h)] for x, y in poly],
            dtype=np.int32
        )
        cv2.fillPoly(mask, [pts], 255)

    return mask


# def snap_mask_to_ink(mask: np.ndarray, ink: np.ndarray) -> np.ndarray:
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     ink_d = cv2.dilate(ink, kernel, iterations=1)
#     return cv2.bitwise_and(mask, ink_d)

def snap_mask_to_ink_connected(polygon_mask: np.ndarray, ink: np.ndarray, *, dilate_seed_px: int = 8) -> np.ndarray:
    """
    Use polygon_mask as a SEED, then keep entire connected ink components that touch the seed.
    This prevents 'cut off half the diagram' when the polygon is slightly too tight.
    """
    # Dilate seed so it touches thin lines reliably
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_seed_px + 1, 2 * dilate_seed_px + 1))
    seed = cv2.dilate(polygon_mask, kernel, iterations=1)

    # Only consider ink pixels
    ink_bin = (ink > 0).astype(np.uint8)

    # Find connected components on ink
    num, labels = cv2.connectedComponents(ink_bin, connectivity=8)

    # Which components are touched by the seed?
    touched = np.unique(labels[(seed > 0) & (ink_bin > 0)])

    # Build output mask as union of those components (skip background label 0)
    out = np.zeros_like(ink_bin, dtype=np.uint8)
    for lab in touched:
        if lab == 0:
            continue
        out[labels == lab] = 255

    return out


def snap_mask_to_ink_connected_bounded(
    polygon_mask: np.ndarray,
    ink: np.ndarray,
    *,
    pad_px: int = 40,            # how far beyond polygon bbox we allow growth
    dilate_seed_px: int = 4,     # smaller than before to avoid touching border
    reject_edge_touch: float = 0.15,  # reject components that touch ROI edges too much
) -> np.ndarray:
    """
    Use polygon_mask as a seed, but only grow within a padded ROI around the polygon.
    This prevents grabbing page border / tables elsewhere.
    """
    H, W = ink.shape[:2]
    ink_bin = (ink > 0).astype(np.uint8)

    # Seed dilation (small!)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_seed_px + 1, 2 * dilate_seed_px + 1))
    seed = cv2.dilate((polygon_mask > 0).astype(np.uint8) * 255, kernel, iterations=1)

    # ROI bounds from polygon mask bbox
    ys, xs = np.where(polygon_mask > 0)
    if len(xs) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    rx1 = max(0, int(x1) - pad_px)
    ry1 = max(0, int(y1) - pad_px)
    rx2 = min(W - 1, int(x2) + pad_px)
    ry2 = min(H - 1, int(y2) + pad_px)

    roi_ink = ink_bin[ry1:ry2+1, rx1:rx2+1]
    roi_seed = (seed[ry1:ry2+1, rx1:rx2+1] > 0).astype(np.uint8)

    # Connected components within ROI
    num, labels = cv2.connectedComponents(roi_ink, connectivity=8)

    # Labels touched by the seed (within ROI)
    touched = np.unique(labels[(roi_seed > 0) & (roi_ink > 0)])

    out_roi = np.zeros_like(roi_ink, dtype=np.uint8)

    # Reject border-like components: those that strongly touch ROI edges
    roi_h, roi_w = roi_ink.shape[:2]
    for lab in touched:
        if lab == 0:
            continue
        comp = (labels == lab)

        # Edge-touch ratio: fraction of comp pixels that lie on ROI boundary
        top = comp[0, :].sum()
        bot = comp[-1, :].sum()
        left = comp[:, 0].sum()
        right = comp[:, -1].sum()
        edge_touch = float(top + bot + left + right) / float(max(1, comp.sum()))

        if edge_touch > reject_edge_touch:
            # likely border/table line hugging ROI boundary
            continue

        out_roi[comp] = 255

    # Place ROI back into full mask
    out = np.zeros((H, W), dtype=np.uint8)
    out[ry1:ry2+1, rx1:rx2+1] = out_roi * 255
    return out



# -----------------------------
# Export PNG with alpha
# -----------------------------

def export_cutout(
    rgb: np.ndarray,
    mask: np.ndarray,
    out_path: Path
):
    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = (mask > 0).astype(np.uint8) * 255

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    cut = rgba[y1:y2+1, x1:x2+1]
    Image.fromarray(cut, "RGBA").save(out_path)

def export_cutout_white_bg(rgb: np.ndarray, mask: np.ndarray, out_path):
    """
    Exports an RGB PNG (no transparency). Pixels outside mask become white.
    """
    mask_bool = (mask > 0)

    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    crop_rgb = rgb[y1:y2+1, x1:x2+1].copy()
    crop_mask = mask_bool[y1:y2+1, x1:x2+1]

    # Paint background white
    crop_rgb[~crop_mask] = 255

    Image.fromarray(crop_rgb, "RGB").save(out_path)



# -----------------------------
# Main pipeline
# -----------------------------

def load_input_image(path: str, dpi: int) -> np.ndarray:
    """
    Load either:
    - an image file (png/jpg/etc)
    - or the first page of a PDF rendered at `dpi`

    Returns RGB numpy array.
    """
    ext = Path(path).suffix.lower()

    if ext in [".png", ".jpg", ".jpeg", ".webp"]:
        return np.array(Image.open(path).convert("RGB"))

    # otherwise assume PDF
    img = render_pdf_page(path, page_index=0, dpi=dpi)
    return np.array(img)

def segment_pdf_with_vlm(pdf_path: str, out_dir: str):
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # --- render for segmentation ---
    seg_np = load_input_image(pdf_path, dpi=SEG_DPI)

    ink = build_ink_no_border(seg_np)

    # --- create preview for VLM ---
    ink_pil = Image.fromarray(ink, "L").convert("RGB")
    ink_pil.thumbnail((INK_PREVIEW_WIDTH, INK_PREVIEW_WIDTH))
    ink_preview_path = out_root / "ink_preview.png"
    ink_pil.save(ink_preview_path)

    # --- call VLM ---
    result = call_vlm_for_polygons(ink_preview_path)

    # --- high-res export ---
    hi_img = render_pdf_page(pdf_path, 0, EXPORT_DPI)
    hi_np = np.array(hi_img)
    ink_hi = build_ink_no_border(hi_np)

    diagrams_dir = out_root / "diagrams"
    diagrams_dir.mkdir(exist_ok=True)

    for d in result.get("diagrams", []):
        poly_mask = polygons_to_mask(
            ink_hi.shape[0],
            ink_hi.shape[1],
            d["polygons"]
        )

        refined = snap_mask_to_ink_connected(poly_mask, ink_hi, dilate_seed_px=10)

        export_cutout_white_bg(hi_np, refined, diagrams_dir / f"{d['id']}.png")


    (out_root / "manifest.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )


# -----------------------------
# CLI entrypoint
# -----------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python segment_diagrams_vlm.py <pdf_path> <out_dir>")
        sys.exit(1)

    segment_pdf_with_vlm(sys.argv[1], sys.argv[2])
