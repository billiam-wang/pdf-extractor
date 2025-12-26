import fitz
import os
import io
import traceback
from PIL import Image
from pathlib import Path
import openai
import pytesseract

from .utils import extract_first_text_response


class TechnicalDrawingParser:

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "databricks")

        # Databricks configuration
        self.databricks_host = "https://dbc-101acd43-457b.cloud.databricks.com"
        self.databricks_token = os.getenv("DATABRICKS_TOKEN", "")
        self.model_name = "databricks-gpt-oss-120b"

    def extract_specifications_with_llm(self, text_content):
        if not text_content or len(text_content.strip()) < 10:
            return "No text content available for specification extraction."

        prompt = f"""Analyze the following technical drawing document and extract specifications in JSON format.

Look for:
- ID number (usually to the left of "drawing no", may not be labeled)
- Electrical properties (voltage, current, resistance, capacitance, ratings, etc.)
- Materials (contact material, insulator material, plating, etc.)
- Operating temperature range

Document text:
{text_content[:10000]}

Return ONLY a JSON object in this exact format:
{{
  "id": "the number to the left of drawing no (or empty string if not found)",
  "specifications": {{
    "electrical": ["list of electrical properties found"],
    "material": ["list of materials and composition found"],
    "operation_temperature": "temperature range if found, empty string otherwise"
  }}
}}

Do not include any explanation, only return the JSON object."""

        try:
            # Use OpenAI client with Databricks endpoint
            client = openai.OpenAI(
                base_url=f"{self.databricks_host}/serving-endpoints",
                api_key=self.databricks_token or "dummy-key"  # Token may not be needed in Databricks Apps
            )

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical document analyzer specializing in extracting specifications."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2048,
                temperature=0.1
            )

            content = response.choices[0].message.content

            # Extract first non-reasoning text response using common utility
            return extract_first_text_response(content)

        except Exception as e:
            return f"Error extracting specifications: {str(e)}\n{traceback.format_exc()}"

    def extract_text_from_image(self, file_path):
        """
        Extract text from an image file using OCR.

        Args:
            file_path: Path to the image file

        Returns:
            Extracted text content
        """
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            print(f"Error extracting text from image {file_path}: {e}")
            return ""

    def extract_images_from_pdf(self, file_path, output_dir=None):
        images = []

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(file_path), "extracted_images")

        os.makedirs(output_dir, exist_ok=True)

        try:
            pdf_document = fitz.open(file_path)
            base_filename = Path(file_path).stem

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images(full=True)

                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]

                    try:
                        # Extract image
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Create filename
                        image_filename = f"{base_filename}_page{page_num + 1}_img{img_index + 1}.{image_ext}"
                        image_path = os.path.join(output_dir, image_filename)

                        # Save image
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        # Get image dimensions
                        try:
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            width, height = pil_image.size
                        except:
                            width, height = 0, 0

                        images.append({
                            "filename": image_filename,
                            "path": image_path,
                            "page": page_num + 1,
                            "index": img_index + 1,
                            "format": image_ext,
                            "width": width,
                            "height": height,
                            "size_bytes": len(image_bytes)
                        })

                    except Exception as e:
                        print(f"Error extracting image {img_index} from page {page_num}: {e}")
                        continue

            pdf_document.close()

        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")

        return images

    def parse(self, file_path):
        result = {
            "filename": os.path.basename(file_path),
            "status": "success",
            "specifications": "",
            "diagrams": [],
            "page_count": 0,
            "error": None
        }

        try:
            # Check file extension to determine file type
            file_ext = Path(file_path).suffix.lower()
            is_image = file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']

            if is_image:
                # Handle image files with OCR
                result["page_count"] = 1  # Images are single-page

                # Extract text using OCR
                full_text = self.extract_text_from_image(file_path)

                # Extract specifications using LLM
                if full_text.strip():
                    result["specifications"] = self.extract_specifications_with_llm(full_text)
                else:
                    result["specifications"] = "No text content found in image."

                # For images, the file itself is the diagram (no need to extract)
                # Optionally, we could copy the image to extracted_images directory
                result["diagrams"] = []

            else:
                # Handle PDF files
                pdf_document = fitz.open(file_path)
                result["page_count"] = len(pdf_document)

                text_parts = []
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    text = page.get_text()
                    if text:
                        text_parts.append(text)

                full_text = "\n".join(text_parts)
                pdf_document.close()

                # Extract specifications using LLM
                if full_text.strip():
                    result["specifications"] = self.extract_specifications_with_llm(full_text)
                else:
                    result["specifications"] = "No text content found in PDF."

                # Extract diagrams/images from PDF
                result["diagrams"] = self.extract_images_from_pdf(file_path)

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()

        return result
