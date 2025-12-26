"""
Technical Drawing Parser

Specialized parser for technical drawings and specification documents.
Extracts:
- Technical specifications (materials, electrical properties, dimensions, etc.)
- Diagrams and images
"""

import fitz  # PyMuPDF
import os
import io
import traceback
from PIL import Image
from pathlib import Path
import openai


class TechnicalDrawingParser:
    """Parser for technical drawing PDFs."""

    def __init__(self, llm_provider=None):
        """
        Initialize the technical drawing parser.

        Args:
            llm_provider: LLM provider to use (databricks)
        """
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "databricks")

        # Databricks configuration
        self.databricks_host = "https://dbc-101acd43-457b.cloud.databricks.com"
        self.databricks_token = os.getenv("DATABRICKS_TOKEN", "")
        self.model_name = "databricks-gpt-oss-120b"

    def extract_specifications_with_llm(self, text_content):
        """
        Extract technical specifications from PDF text using an LLM.

        Args:
            text_content: Full text content from the PDF

        Returns:
            Extracted specifications text or error message
        """
        if not text_content or len(text_content.strip()) < 10:
            return "No text content available for specification extraction."

        prompt = f"""Analyze the following technical document text and extract all technical specifications.

Focus on extracting:
- Materials and composition
- Electrical properties (voltage, current, resistance, capacitance, etc.)
- Physical dimensions and measurements
- Performance characteristics
- Operating conditions (temperature, pressure, etc.)
- Standards and certifications
- Part numbers and model information

If no specifications are found, respond with "No technical specifications found in this document."

Document text:
{text_content[:10000]}

Please provide a clear, organized summary of the technical specifications:"""

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

            # Parse response to extract only non-reasoning text
            # Handle structured responses that may contain reasoning blocks
            try:
                import json
                parsed = json.loads(content)

                # Extract first non-reasoning text block
                if isinstance(parsed, dict):
                    if parsed.get('type') == 'text':
                        return parsed.get('text', content)
                    elif 'text' in parsed:
                        return parsed['text']
                elif isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            return item.get('text', content)

                # If we couldn't parse structured format, return original
                return content
            except (json.JSONDecodeError, TypeError):
                # Not JSON, return as-is
                return content

        except Exception as e:
            return f"Error extracting specifications: {str(e)}\n{traceback.format_exc()}"

    def extract_images_from_pdf(self, file_path, output_dir=None):
        """
        Extract all images/diagrams from a PDF file.

        Args:
            file_path: Path to the PDF file
            output_dir: Directory to save extracted images (optional)

        Returns:
            List of dictionaries containing image information
        """
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
        """
        Parse a technical drawing PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary with: filename, status, specifications, diagrams, page_count, error
        """
        result = {
            "filename": os.path.basename(file_path),
            "status": "success",
            "specifications": "",
            "diagrams": [],
            "page_count": 0,
            "error": None
        }

        try:
            # Extract text content
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

            # Extract diagrams/images
            result["diagrams"] = self.extract_images_from_pdf(file_path)

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()

        return result
