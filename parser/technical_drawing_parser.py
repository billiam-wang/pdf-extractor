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
import anthropic
import openai


class TechnicalDrawingParser:
    """Parser for technical drawing PDFs."""

    def __init__(self, llm_provider=None):
        """
        Initialize the technical drawing parser.

        Args:
            llm_provider: LLM provider to use (anthropic, openai, databricks)
        """
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "databricks")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

        # Databricks configuration
        self.databricks_host = os.getenv("DATABRICKS_HOST", "")
        self.databricks_token = os.getenv("DATABRICKS_TOKEN", "")
        self.databricks_endpoint = os.getenv("DATABRICKS_SERVING_ENDPOINT", "")

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
                return self._call_databricks_endpoint(prompt)

            else:
                return f"Error: Unknown LLM provider '{self.llm_provider}'. Supported: anthropic, openai, databricks"

        except Exception as e:
            return f"Error extracting specifications with {self.llm_provider}: {str(e)}\n{traceback.format_exc()}"

    def _call_databricks_endpoint(self, prompt):
        """
        Call Databricks model serving endpoint.

        Supports both:
        1. Databricks Apps (automatic workspace authentication)
        2. External calls (using DATABRICKS_HOST and DATABRICKS_TOKEN)
        """
        import requests

        # Determine endpoint URL
        if self.databricks_endpoint:
            # Full endpoint URL provided
            endpoint_url = self.databricks_endpoint
        elif self.databricks_host:
            # Build URL from host - you'll need to set the serving endpoint name
            endpoint_name = os.getenv("DATABRICKS_ENDPOINT_NAME", "")
            if not endpoint_name:
                return "Error: DATABRICKS_ENDPOINT_NAME not set. Please set the serving endpoint name."
            endpoint_url = f"{self.databricks_host}/serving-endpoints/{endpoint_name}/invocations"
        else:
            return "Error: DATABRICKS_HOST or DATABRICKS_SERVING_ENDPOINT must be set."

        # Set up authentication
        headers = {
            "Content-Type": "application/json"
        }

        # Add token if available (for external calls)
        # When running in Databricks Apps, workspace auth is automatic
        if self.databricks_token:
            headers["Authorization"] = f"Bearer {self.databricks_token}"

        # Prepare request payload
        # Adjust this based on your model's expected format
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a technical document analyzer specializing in extracting specifications."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.1
        }

        try:
            response = requests.post(endpoint_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()

            # Extract response - adjust based on your model's response format
            # Common formats:
            if "choices" in result:
                # OpenAI-compatible format
                return result["choices"][0]["message"]["content"]
            elif "predictions" in result:
                # Databricks MLflow format
                return result["predictions"][0]
            elif "content" in result:
                # Direct content format
                return result["content"]
            else:
                # Return full response if format unknown
                return str(result)

        except requests.exceptions.RequestException as e:
            return f"Error calling Databricks endpoint: {str(e)}\nURL: {endpoint_url}"
        except Exception as e:
            return f"Error processing Databricks response: {str(e)}"

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
