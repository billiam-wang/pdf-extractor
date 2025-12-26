import os
import traceback
import json
import io
import uuid
import fitz
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional
import openai

from utils.llm import responses_text
from utils.image import (
    preprocess_drawing_image,
    build_ink_no_border,
    polygons_to_mask,
    snap_mask_to_ink_connected,
    export_cutout_white_bg,
)
from utils.file import encode_file_base64


class TechnicalDrawingParser:
    SEG_DPI = 220
    EXPORT_DPI = 400
    INK_PREVIEW_WIDTH = 1024

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "databricks")
        self.databricks_host = "https://dbc-101acd43-457b.cloud.databricks.com"
        self.databricks_token = os.getenv("DATABRICKS_TOKEN", "")
        self.model_name = "billiamtesting-openai"

    def _render_pdf_page(self, pdf_path: str, page_index: int, dpi: int) -> Image.Image:
        doc = fitz.open(pdf_path)
        try:
            page = doc.load_page(page_index)
            zoom = dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        finally:
            doc.close()

    def _load_input_image(self, path: str, dpi: int) -> np.ndarray:
        ext = Path(path).suffix.lower()

        if ext in [".png", ".jpg", ".jpeg", ".webp"]:
            return np.array(Image.open(path).convert("RGB"))

        img = self._render_pdf_page(path, page_index=0, dpi=dpi)
        return np.array(img)

    def extract_specifications_with_vision(self, file_path):
        prompt = """Analyze this technical drawing document and extract specifications.

Look for:
- ID number (usually to the left of "drawing no", may not be labeled)
- Electrical properties (voltage, current, resistance, capacitance, ratings, etc.)
- Materials (contact material, insulator material, plating, etc.)
- Operating temperature range"""

        response_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The number to the left of 'Drawing No.' or empty if not found. Example ID is 0.5FPCRXXTB4SDP-A"
                },
                "specifications": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "electrical": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of electrical properties found"
                        },
                        "material": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of materials and composition found"
                        },
                        "operation_temperature": {
                            "type": "string",
                            "description": "Temperature range if found, empty string otherwise"
                        }
                    },
                    "required": ["electrical", "material", "operation_temperature"]
                }
            },
            "required": ["id", "specifications"]
        }

        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext == '.pdf':
                pil_img = self._render_pdf_page(file_path, 0, 200)
            else:
                pil_img = Image.open(file_path)

            pil_img = preprocess_drawing_image(pil_img)

            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            base64_file = __import__('base64').b64encode(img_byte_arr.getvalue()).decode('utf-8')

            client = openai.OpenAI(
                base_url=f"{self.databricks_host}/serving-endpoints",
                api_key=self.databricks_token or "dummy-key"
            )

            response = client.responses.create(
                model=self.model_name,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_file}"}
                    ]
                }],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "technical_specifications",
                        "schema": response_schema,
                        "strict": True
                    }
                },
                max_output_tokens=2048,
                temperature=0.1
            )

            return responses_text(response)

        except Exception as e:
            return f"Error extracting specifications: {str(e)}\n{traceback.format_exc()}"

    def extract_diagrams_with_vision(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        max_pages: int = 1,
    ) -> List[Dict[str, Any]]:

        prompt = """Return ONLY valid JSON. No markdown. No extra keys. No explanations.

You are given a black-and-white technical drawing page (line art only). The page contains MULTIPLE separate diagrams/views (e.g., top view, side view, section view, isometric view), plus tables/spec blocks/title blocks.

GOAL:
Extract EACH individual diagram/view as its own item. DO NOT combine multiple views into one output item.

DEFINITIONS:
- "One diagram/view" = one logical illustration of the part in a single view or depiction.
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

EXCLUDE COMPLETELY:
- tables (dimension tables)
- specification text blocks
- title blocks / logos / contact info
- page border/frame
- watermarks

FINAL CHECK BEFORE YOU RESPOND:
- Each "diagram_i" contains ONLY ONE view/depiction.
- If any diagram includes parts of two views, split it into two diagrams.
- Prefer more diagrams rather than fewer.
"""

        response_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "diagrams": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string"},
                            "polygons": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 2,
                                        "maxItems": 2
                                    }
                                }
                            }
                        },
                        "required": ["id", "polygons"]
                    }
                }
            },
            "required": ["diagrams"]
        }

        def _ensure_output_dir() -> str:
            if output_dir:
                out = output_dir
            else:
                p = Path(file_path)
                out = str(p.parent / f"{p.stem}_diagrams")
            Path(out).mkdir(parents=True, exist_ok=True)
            return out

        out_dir = _ensure_output_dir()

        seg_np = self._load_input_image(file_path, dpi=self.SEG_DPI)
        seg_pil = Image.fromarray(seg_np)
        seg_pil = preprocess_drawing_image(seg_pil)
        seg_np = np.array(seg_pil)
        ink = build_ink_no_border(seg_np)

        ink_pil = Image.fromarray(ink, "L").convert("RGB")
        ink_pil.thumbnail((self.INK_PREVIEW_WIDTH, self.INK_PREVIEW_WIDTH))
        ink_preview_path = Path(out_dir) / "ink_preview.png"
        ink_pil.save(ink_preview_path)

        with open(ink_preview_path, "rb") as f:
            b64 = __import__('base64').b64encode(f.read()).decode("utf-8")

        image_data_url = f"data:image/png;base64,{b64}"

        client = openai.OpenAI(
            base_url=f"{self.databricks_host}/serving-endpoints",
            api_key=self.databricks_token or "dummy-key"
        )

        response = client.responses.create(
            model=self.model_name,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }],
            max_output_tokens=2500,
        )

        raw = responses_text(response)
        payload = json.loads(raw)

        hi_img = self._render_pdf_page(file_path, 0, self.EXPORT_DPI) if Path(file_path).suffix.lower() == '.pdf' else Image.open(file_path)
        hi_img = preprocess_drawing_image(hi_img)
        hi_np = np.array(hi_img)
        ink_hi = build_ink_no_border(hi_np)

        results: List[Dict[str, Any]] = []

        for d in payload.get("diagrams", []):
            poly_mask = polygons_to_mask(
                ink_hi.shape[0],
                ink_hi.shape[1],
                d["polygons"]
            )

            refined = snap_mask_to_ink_connected(poly_mask, ink_hi, dilate_seed_px=10)

            fname = f"{d['id']}_{uuid.uuid4().hex[:8]}.png"
            img_path = str(Path(out_dir) / fname)

            export_cutout_white_bg(hi_np, refined, img_path)

            results.append({
                "filename": fname,
                "page": 1,
                "path": os.path.abspath(img_path),
                "description": d['id'],
            })

        return results

    def parse(self, file_path):
        result = {
            "filename": os.path.basename(file_path),
            "status": "success",
            "specifications": "",
            "diagrams": [],
            "page_count": 1,
            "error": None
        }

        try:
            result["specifications"] = self.extract_specifications_with_vision(file_path)
            result["diagrams"] = self.extract_diagrams_with_vision(file_path)

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()

        return result
