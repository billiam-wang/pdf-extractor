import os
import traceback
import base64
from pathlib import Path
import openai

from .utils import extract_first_text_response


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
        """
        Extract specifications from a document (PDF or image) using vision API.

        Args:
            file_path: Path to document file (PDF, PNG, JPG, etc.)

        Returns:
            JSON string with extracted specifications
        """
        prompt = """Analyze this technical drawing document and extract specifications.

Look for:
- ID number (usually to the left of "drawing no", may not be labeled)
- Electrical properties (voltage, current, resistance, capacitance, ratings, etc.)
- Materials (contact material, insulator material, plating, etc.)
- Operating temperature range"""

        # Define the response schema
        response_schema = {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The number to the left of drawing no (or empty string if not found)"
                },
                "specifications": {
                    "type": "object",
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
            # Encode file to base64
            base64_file = self.encode_file(file_path)

            # Determine media type based on file extension
            file_ext = Path(file_path).suffix.lower()
            media_type = "application/pdf" if file_ext == '.pdf' else "image/jpeg"

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
            import json
            content = response.choices[0].message.content

            # The content should already be valid JSON matching our schema
            return content

        except Exception as e:
            return f"Error extracting specifications: {str(e)}\n{traceback.format_exc()}"


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
            # Pass file directly to vision API (handles both PDFs and images)
            result["specifications"] = self.extract_specifications_with_vision(file_path)

            # No diagram extraction - vision API processes the entire document
            result["diagrams"] = []

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()

        return result
