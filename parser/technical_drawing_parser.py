import os
import traceback
import base64
from pathlib import Path
import openai
from pdf2image import convert_from_path
from PIL import Image
import io
import json
import uuid
from typing import List, Dict, Any, Optional
import numpy as np

from .utils import responses_text
from collections import deque

def preprocess_drawing_image(pil_img: Image.Image) -> Image.Image:
    """
    1) Replace transparency with white
    2) Replace yellow-ish pixels with white (keeps black/blue lines)
    """
    # --- (1) Flatten transparency onto white ---
    if pil_img.mode in ("RGBA", "LA") or ("transparency" in pil_img.info):
        rgba = pil_img.convert("RGBA")
        white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgba = Image.alpha_composite(white_bg, rgba)
        pil_img = rgba.convert("RGB")
    else:
        pil_img = pil_img.convert("RGB")

    # --- (2) Remove yellow coloring ---
    arr = np.array(pil_img)  # shape (H, W, 3), uint8, RGB
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)

    # Heuristic for "yellow-ish" in RGB:
    # yellow has high R and G, low-ish B.
    # also often bright (watermark tends to be light yellow).
    yellowish = (r > 160) & (g > 160) & (b < 170) & ((r - b) > 40) & ((g - b) > 40)

    # If you want to be more conservative and only remove lighter yellow (watermark-like):
    # bright = ((r + g + b) // 3) > 170
    mask = yellowish # & bright

    # Turn those pixels white
    arr[mask] = [255, 255, 255]

    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


class TechnicalDrawingParser:

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "databricks")

        # Databricks configuration
        self.databricks_host = "https://dbc-101acd43-457b.cloud.databricks.com"
        self.databricks_token = os.getenv("DATABRICKS_TOKEN", "")
        self.model_name = "billiamtesting-openai"

    def encode_file(self, file_path):
        """Encode file (image or PDF) to base64 string."""
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode('utf-8')

    def extract_specifications_with_vision(self, file_path):
        prompt = """Analyze this technical drawing document and extract specifications.

Look for:
- ID number (usually to the left of "drawing no", may not be labeled)
- Electrical properties (voltage, current, resistance, capacitance, ratings, etc.)
- Materials (contact material, insulator material, plating, etc.)
- Operating temperature range"""

        # Define the response schema
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
            # Check if file is PDF and convert to image if needed
            file_ext = Path(file_path).suffix.lower()

            if file_ext == '.pdf':
                # Convert PDF first page to image
                images = convert_from_path(file_path, first_page=1, last_page=1, dpi=200, fmt="png", use_pdftocairo=True)

                if not images:
                    return f"Error: Could not convert PDF to image"

                # Convert PIL image to base64
                img_byte_arr = io.BytesIO()
                images[0].save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                base64_file = base64.b64encode(img_byte_arr.read()).decode('utf-8')
                media_type = "image/jpeg"
            else:
                # For images, encode directly
                base64_file = self.encode_file(file_path)
                media_type = "image/jpeg"

            # Use OpenAI client with Databricks endpoint
            client = openai.OpenAI(
                base_url=f"{self.databricks_host}/serving-endpoints",
                api_key=self.databricks_token or "dummy-key"
            )

            response = client.responses.create(
                model=self.model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:{media_type};base64,{base64_file}"
                            }
                        ]
                    }
                ],
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

            # Parse response - Responses API returns structured data
            content = responses_text(response)

            # The content should already be valid JSON matching our schema
            return content

        except Exception as e:
            return f"Error extracting specifications: {str(e)}\n{traceback.format_exc()}"


    def extract_diagrams_with_vision(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        max_pages: int = 1,
        dpi: int = 200,
        pad: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Uses the LLM to locate diagram panels on each page and crops them into individual images.

        Returns: list of dicts like:
        {
            "page": 1,
            "label": "top",
            "description": "Top view with pin layout and dimensions",
            "bbox": {"x":..., "y":..., "w":..., "h":...},
            "image_path": "/path/to/crop.png"
        }
        """
        prompt = """You are given ONE rendered page image from an engineering drawing.
    Find each distinct DIAGRAM panel (e.g., isometric/top/front/side/section/detail).
    Return generous and reasonable bounding boxes around each diagram panel’s content.

    Rules:
    - Do NOT merge multiple diagrams into one box.
    - Bounding boxes should include the entire diagram. Do not cutoff any part of the main diagram and include all dimension lines, leader lines, arrows, and tolerances directly associated with the diagram.
    - Exclude title blocks / company info / footer unless it contains an actual diagram.
    - Exclude general specification text, only capture text if it is directly related to the current diagram panel.
    - Coordinates must be pixel-based relative to this image: x,y,w,h with x,y at top-left.
    - label should be 1–3 words chosen from: isometric, top, front, side, section, detail, other.
    - description must be very brief (5–10 words) and not guess dimensions/part numbers.
    - Return boxes sorted top-to-bottom, left-to-right.
    """

        # Strict schema: additionalProperties must be false at every object node.
        response_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "page": {"type": "integer"},
                "image_width": {"type": "integer"},
                "image_height": {"type": "integer"},
                "diagrams": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": {"type": "string"},
                            "description": {"type": "string"},
                            "bbox": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "x": {"type": "integer"},
                                    "y": {"type": "integer"},
                                    "w": {"type": "integer"},
                                    "h": {"type": "integer"},
                                },
                                "required": ["x", "y", "w", "h"],
                            },
                        },
                        "required": ["label", "description", "bbox"],
                    },
                },
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["page", "image_width", "image_height", "diagrams", "warnings"],
        }

        def _clamp_bbox(b: Dict[str, int], W: int, H: int, pad_px: int) -> Dict[str, int]:
            x1 = max(0, int(b["x"]) - pad_px)
            y1 = max(0, int(b["y"]) - pad_px)
            x2 = min(W, int(b["x"]) + int(b["w"]) + pad_px)
            y2 = min(H, int(b["y"]) + int(b["h"]) + pad_px)
            return {"x": x1, "y": y1, "w": max(1, x2 - x1), "h": max(1, y2 - y1)}

        def _ensure_output_dir() -> str:
            # Default: sibling folder next to the input
            if output_dir:
                out = output_dir
            else:
                p = Path(file_path)
                out = str(p.parent / f"{p.stem}_diagrams")
            Path(out).mkdir(parents=True, exist_ok=True)
            return out

        out_dir = _ensure_output_dir()

        # Render pages to PIL images
        file_ext = Path(file_path).suffix.lower()
        page_images: List[Image.Image] = []

        if file_ext == ".pdf":
            # Convert up to max_pages
            page_images = convert_from_path(file_path, dpi=dpi, first_page=1, last_page=max_pages, fmt="png", use_pdftocairo=True)
        else:
            page_images = [Image.open(file_path).convert("RGBA")]

        # OpenAI client with Databricks endpoint (same as your specs)
        client = openai.OpenAI(
            base_url=f"{self.databricks_host}/serving-endpoints",
            api_key=self.databricks_token or "dummy-key"
        )

        results: List[Dict[str, Any]] = []

        for page_idx, pil_img in enumerate(page_images, start=1):
            pil_img = preprocess_drawing_image(pil_img)
            # Encode page image as JPEG base64 for vision
            img_byte_arr = io.BytesIO()
            pil_img.convert("RGB").save(img_byte_arr, format="JPEG", quality=90)
            img_byte_arr.seek(0)
            base64_file = base64.b64encode(img_byte_arr.read()).decode("utf-8")
            media_type = "image/jpeg"

            W, H = pil_img.size

            response = client.responses.create(
                model=self.model_name,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_text", "text": f"Page={page_idx}, image_size={W}x{H} pixels."},
                        {"type": "input_image", "image_url": f"data:{media_type};base64,{base64_file}"},
                    ]
                }],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "diagram_extraction",
                        "schema": response_schema,
                        "strict": True
                    }
                },
                max_output_tokens=1200,
                temperature=0.1,
            )

            # Parse content robustly using your helper (returns the JSON string)
            content = responses_text(response)
            payload = json.loads(content)

            full_bbox = {"x": 0, "y": 0, "w": W, "h": H}
            preprocessed_drawing_image = {"bbox": full_bbox, "label": "", "description": "Preprocessed original diagram"}

            diagram_specs = [preprocessed_drawing_image] + payload["diagrams"]

            for i, d in enumerate(diagram_specs):
                pad = max(12, int(0.2 * max(d["bbox"]["w"], d["bbox"]["h"])))
                bbox = _clamp_bbox(d["bbox"], W, H, pad_px=pad)
                crop = pil_img.crop((bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]))

                safe_label = "".join(ch if ch.isalnum() else "_" for ch in d["label"].lower()).strip("_") or "diagram"
                fname = f"p{page_idx:02d}_{i:02d}_{safe_label}_{uuid.uuid4().hex[:8]}.png"
                img_path = str(Path(out_dir) / fname)
                crop.save(img_path, format="PNG")

                results.append({
                    "filename": fname,
                    "page": page_idx,
                    "width": crop.size[0],
                    "height": crop.size[1],
                    "path": os.path.abspath(img_path),
                    "label": d["label"],
                    "description": d["description"],
                    "bbox": bbox,
                })

        return results


    def parse(self, file_path):
        result = {
            "filename": os.path.basename(file_path),
            "status": "success",
            "specifications": "",
            "diagrams": [],
            "page_count": 1,  # Default to 1 for images, PDFs handled by vision API
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
