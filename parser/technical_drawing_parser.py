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
from typing import Optional
from PIL import Image
from pathlib import Path
import anthropic
import openai


class TechnicalDrawingParser:
    """Parser for technical drawing PDFs."""

    def __init__(self, llm_provider: Optional[str] = None):
        """
        Initialize the technical drawing parser.

        Args:
            llm_provider: LLM provider to use (anthropic, openai, databricks)
        """
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "anthropic")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.databricks_api_url = os.getenv("DATABRICKS_API_URL", "")
        self.databricks_token = os.getenv("DATABRICKS_TOKEN", "")

    def extract_specifications_with_llm(self, text_content: str) -> str:
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
            if self.llm_provider == "anthropic":
                if not self.anthropic_api_key:
                    return "Error: ANTHROPIC_API_KEY not set. Please configure your API key."

                client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                message = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2048,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return message.content[0].text

            elif self.llm_provider == "openai":
                if not self.openai_api_key:
                    return "Error: OPENAI_API_KEY not set. Please configure your API key."

                client = openai.OpenAI(api_key=self.openai_api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a technical document analyzer specializing in extracting specifications."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2048
                )
                return response.choices[0].message.content

            elif self.llm_provider == "databricks":
                if not self.databricks_api_url or not self.databricks_token:
                    return "Error: DATABRICKS_API_URL and DATABRICKS_TOKEN must be set for Databricks provider."

                import requests
                headers = {
                    "Authorization": f"Bearer {self.databricks_token}",
                    "Content-Type": "application/json"
                }
                data = {
                    "messages": [
                        {"role": "system", "content": "You are a technical document analyzer specializing in extracting specifications."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 2048
                }

                response = requests.post(self.databricks_api_url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "No response from model")

            else:
                return f"Error: Unknown LLM provider '{self.llm_provider}'. Supported: anthropic, openai, databricks"

        except Exception as e:
            return f"Error extracting specifications with {self.llm_provider}: {str(e)}\n{traceback.format_exc()}"

    def extract_images_from_pdf(self, file_path: str, output_dir: Optional[str] = None):
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

    def parse(self, file_path: str):
        """
        Parse a technical drawing PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary containing parsed information
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
