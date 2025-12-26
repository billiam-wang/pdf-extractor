import cv2
import numpy as np
from PIL import Image
from typing import List
from pathlib import Path


def preprocess_drawing_image(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode in ("RGBA", "LA") or ("transparency" in pil_img.info):
        rgba = pil_img.convert("RGBA")
        white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgba = Image.alpha_composite(white_bg, rgba)
        pil_img = rgba.convert("RGB")
    else:
        pil_img = pil_img.convert("RGB")

    arr = np.array(pil_img)
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)

    yellowish = (r > 160) & (g > 160) & (b < 170) & ((r - b) > 40) & ((g - b) > 40)
    mask = yellowish
    arr[mask] = [255, 255, 255]

    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


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

    h, w = ink.shape
    margin = 10
    ink[:margin, :] = 0
    ink[-margin:, :] = 0
    ink[:, :margin] = 0
    ink[:, -margin:] = 0

    return ink


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


def snap_mask_to_ink_connected(polygon_mask: np.ndarray, ink: np.ndarray, *, dilate_seed_px: int = 8) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_seed_px + 1, 2 * dilate_seed_px + 1))
    seed = cv2.dilate(polygon_mask, kernel, iterations=1)

    ink_bin = (ink > 0).astype(np.uint8)

    num, labels = cv2.connectedComponents(ink_bin, connectivity=8)

    touched = np.unique(labels[(seed > 0) & (ink_bin > 0)])

    out = np.zeros_like(ink_bin, dtype=np.uint8)
    for lab in touched:
        if lab == 0:
            continue
        out[labels == lab] = 255

    return out


def export_cutout_white_bg(rgb: np.ndarray, mask: np.ndarray, out_path):
    mask_bool = (mask > 0)

    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    crop_rgb = rgb[y1:y2+1, x1:x2+1].copy()
    crop_mask = mask_bool[y1:y2+1, x1:x2+1]

    crop_rgb[~crop_mask] = 255

    Image.fromarray(crop_rgb, "RGB").save(out_path)
