import gradio as gr
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import json
import traceback
import os
import io
from PIL import Image
import base64
from pathlib import Path
import anthropic
import openai


# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # Options: "anthropic", "openai", "databricks"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATABRICKS_API_URL = os.getenv("DATABRICKS_API_URL", "")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")


def extract_specifications_with_llm(text_content: str, llm_provider: str = None) -> str:
    """
    Extract technical specifications from PDF text using an LLM.

    Args:
        text_content: Full text content from the PDF
        llm_provider: LLM provider to use (anthropic, openai, databricks)

    Returns:
        Extracted specifications text or error message
    """
    if not text_content or len(text_content.strip()) < 10:
        return "No text content available for specification extraction."

    provider = llm_provider or LLM_PROVIDER

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
        if provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                return "Error: ANTHROPIC_API_KEY not set. Please configure your API key."

            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text

        elif provider == "openai":
            if not OPENAI_API_KEY:
                return "Error: OPENAI_API_KEY not set. Please configure your API key."

            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a technical document analyzer specializing in extracting specifications."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048
            )
            return response.choices[0].message.content

        elif provider == "databricks":
            if not DATABRICKS_API_URL or not DATABRICKS_TOKEN:
                return "Error: DATABRICKS_API_URL and DATABRICKS_TOKEN must be set for Databricks provider."

            import requests
            headers = {
                "Authorization": f"Bearer {DATABRICKS_TOKEN}",
                "Content-Type": "application/json"
            }
            data = {
                "messages": [
                    {"role": "system", "content": "You are a technical document analyzer specializing in extracting specifications."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2048
            }

            response = requests.post(DATABRICKS_API_URL, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "No response from model")

        else:
            return f"Error: Unknown LLM provider '{provider}'. Supported: anthropic, openai, databricks"

    except Exception as e:
        return f"Error extracting specifications with {provider}: {str(e)}\n{traceback.format_exc()}"


def extract_images_from_pdf(file_path: str, output_dir: str = None) -> List[dict]:
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


def extract_pdf_info(file_path: str, llm_provider: str = None) -> dict:
    """
    Extract specifications and diagrams from a single PDF file.

    Args:
        file_path: Path to the PDF file
        llm_provider: LLM provider to use for specification extraction

    Returns:
        Dictionary containing extracted information
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
            result["specifications"] = extract_specifications_with_llm(full_text, llm_provider)
        else:
            result["specifications"] = "No text content found in PDF."

        # Extract diagrams/images
        result["diagrams"] = extract_images_from_pdf(file_path)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


def process_pdfs_parallel(files: List[str], max_workers: int = 4, llm_provider: str = None) -> Tuple[str, str, List[str]]:
    """
    Process multiple PDF files in parallel.

    Args:
        files: List of file paths
        max_workers: Maximum number of parallel workers
        llm_provider: LLM provider to use

    Returns:
        Tuple of (summary_text, detailed_json, diagram_paths)
    """
    if not files:
        return "No files uploaded.", "{}", []

    results = []
    all_diagrams = []

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(extract_pdf_info, file, llm_provider): file for file in files}

        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)

                # Collect all diagram paths
                for diagram in result.get("diagrams", []):
                    all_diagrams.append(diagram["path"])

            except Exception as e:
                results.append({
                    "filename": os.path.basename(file_path),
                    "status": "error",
                    "error": f"Processing failed: {str(e)}",
                    "specifications": "",
                    "diagrams": [],
                    "page_count": 0
                })

    # Generate summary
    summary_parts = [f"Processed {len(results)} file(s):\n"]
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = len(results) - success_count
    total_diagrams = sum(len(r.get("diagrams", [])) for r in results)

    summary_parts.append(f"✅ Successful: {success_count}")
    summary_parts.append(f"❌ Errors: {error_count}")
    summary_parts.append(f"🖼️  Total diagrams extracted: {total_diagrams}\n")

    for i, result in enumerate(results, 1):
        summary_parts.append(f"\n{'='*80}")
        summary_parts.append(f"File {i}: {result['filename']}")
        summary_parts.append(f"Status: {result['status'].upper()}")

        if result["status"] == "success":
            summary_parts.append(f"Pages: {result['page_count']}")
            summary_parts.append(f"Diagrams extracted: {len(result['diagrams'])}")

            summary_parts.append("\n--- TECHNICAL SPECIFICATIONS ---")
            summary_parts.append(result.get("specifications", "No specifications found"))

            if result["diagrams"]:
                summary_parts.append("\n--- EXTRACTED DIAGRAMS ---")
                for diagram in result["diagrams"]:
                    summary_parts.append(
                        f"  • {diagram['filename']} - Page {diagram['page']}, "
                        f"{diagram['width']}x{diagram['height']}px, "
                        f"{diagram['size_bytes'] / 1024:.1f} KB"
                    )
        else:
            summary_parts.append(f"Error: {result['error']}")

    summary_text = "\n".join(summary_parts)
    detailed_json = json.dumps(results, indent=2, ensure_ascii=False, default=str)

    return summary_text, detailed_json, all_diagrams


def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(title="PDF Technical Specification & Diagram Extractor", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🔧 PDF Technical Specification & Diagram Extractor

            Upload technical PDF documents to automatically extract:
            - **Technical Specifications** (materials, electrical properties, dimensions, etc.)
            - **Diagrams & Images** (all diagrams isolated as separate image files)

            Powered by AI for intelligent specification extraction.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")

                file_input = gr.File(
                    label="Upload PDF Files",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath"
                )

                llm_provider = gr.Dropdown(
                    choices=["anthropic", "openai", "databricks"],
                    value=LLM_PROVIDER,
                    label="LLM Provider",
                    info="Choose the AI model provider for specification extraction"
                )

                max_workers = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=4,
                    step=1,
                    label="Parallel Workers",
                    info="Number of files to process simultaneously"
                )

                process_btn = gr.Button("🚀 Extract Specifications & Diagrams", variant="primary", size="lg")

        gr.Markdown("### 📊 Results")

        with gr.Tabs():
            with gr.Tab("📝 Specifications Summary"):
                summary_output = gr.Textbox(
                    label="Extraction Summary",
                    lines=25,
                    max_lines=40,
                    show_copy_button=True
                )

            with gr.Tab("🖼️  Extracted Diagrams"):
                diagram_gallery = gr.Gallery(
                    label="Diagrams & Images",
                    show_label=True,
                    columns=3,
                    height="auto",
                    object_fit="contain"
                )

            with gr.Tab("📋 Detailed JSON"):
                json_output = gr.Code(
                    label="Complete Results (JSON)",
                    language="json",
                    lines=20
                )

        gr.Markdown(
            """
            ---
            ### 🎯 How to use:
            1. **Configure LLM**: Set your API key as environment variable (ANTHROPIC_API_KEY, OPENAI_API_KEY, or DATABRICKS credentials)
            2. **Upload PDFs**: Click or drag-and-drop one or more technical PDF files
            3. **Select Provider**: Choose your preferred AI model provider
            4. **Adjust Workers**: Set parallel processing level (default: 4)
            5. **Extract**: Click the extract button and wait for processing
            6. **Review Results**:
               - **Specifications Summary**: AI-extracted technical specs
               - **Extracted Diagrams**: Visual gallery of all diagrams
               - **Detailed JSON**: Complete structured data

            ### 🔑 Environment Variables:
            ```bash
            # For Anthropic (Claude)
            export ANTHROPIC_API_KEY="your-api-key"

            # For OpenAI (GPT-4)
            export OPENAI_API_KEY="your-api-key"

            # For Databricks
            export DATABRICKS_API_URL="your-model-endpoint-url"
            export DATABRICKS_TOKEN="your-access-token"
            ```

            **Note**: Diagrams are saved to `extracted_images/` directory.
            """
        )

        # Wire up the event handler
        process_btn.click(
            fn=process_pdfs_parallel,
            inputs=[file_input, max_workers, llm_provider],
            outputs=[summary_output, json_output, diagram_gallery]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
