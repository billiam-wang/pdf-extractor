import gradio as gr
import PyPDF2
import pdfplumber
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import json
import traceback


def extract_pdf_info(file_path: str) -> dict:
    """
    Extract information from a single PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Dictionary containing extracted information
    """
    result = {
        "filename": file_path.split("/")[-1],
        "status": "success",
        "text_content": "",
        "metadata": {},
        "page_count": 0,
        "error": None
    }

    try:
        # Extract text using pdfplumber
        with pdfplumber.open(file_path) as pdf:
            result["page_count"] = len(pdf.pages)
            text_parts = []

            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {page_num} ---\n{page_text}\n")

            result["text_content"] = "\n".join(text_parts)

        # Extract metadata using PyPDF2
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = pdf_reader.metadata

            if metadata:
                result["metadata"] = {
                    "title": metadata.get('/Title', 'N/A'),
                    "author": metadata.get('/Author', 'N/A'),
                    "subject": metadata.get('/Subject', 'N/A'),
                    "creator": metadata.get('/Creator', 'N/A'),
                    "producer": metadata.get('/Producer', 'N/A'),
                    "creation_date": metadata.get('/CreationDate', 'N/A'),
                }

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


def process_pdfs_parallel(files: List[str], max_workers: int = 4) -> Tuple[str, str]:
    """
    Process multiple PDF files in parallel.

    Args:
        files: List of file paths
        max_workers: Maximum number of parallel workers

    Returns:
        Tuple of (summary_text, detailed_json)
    """
    if not files:
        return "No files uploaded.", "{}"

    results = []

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(extract_pdf_info, file): file for file in files}

        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    "filename": file_path.split("/")[-1],
                    "status": "error",
                    "error": f"Processing failed: {str(e)}",
                    "text_content": "",
                    "metadata": {},
                    "page_count": 0
                })

    # Generate summary
    summary_parts = [f"Processed {len(results)} file(s):\n"]
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = len(results) - success_count

    summary_parts.append(f"✅ Successful: {success_count}")
    summary_parts.append(f"❌ Errors: {error_count}\n")

    for i, result in enumerate(results, 1):
        summary_parts.append(f"\n{'='*60}")
        summary_parts.append(f"File {i}: {result['filename']}")
        summary_parts.append(f"Status: {result['status'].upper()}")

        if result["status"] == "success":
            summary_parts.append(f"Pages: {result['page_count']}")
            summary_parts.append(f"Text length: {len(result['text_content'])} characters")

            if result["metadata"]:
                summary_parts.append("\nMetadata:")
                for key, value in result["metadata"].items():
                    summary_parts.append(f"  {key}: {value}")

            # Show first 500 characters of extracted text
            if result["text_content"]:
                preview = result["text_content"][:500]
                if len(result["text_content"]) > 500:
                    preview += "..."
                summary_parts.append(f"\nText Preview:\n{preview}")
        else:
            summary_parts.append(f"Error: {result['error']}")

    summary_text = "\n".join(summary_parts)
    detailed_json = json.dumps(results, indent=2, ensure_ascii=False)

    return summary_text, detailed_json


def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(title="PDF Information Extractor") as demo:
        gr.Markdown(
            """
            # 📄 PDF Information Extractor

            Upload one or more PDF files to extract text content and metadata.
            Files are processed in parallel for faster results.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload PDF Files",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath"
                )

                max_workers = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=4,
                    step=1,
                    label="Parallel Workers",
                    info="Number of files to process simultaneously"
                )

                process_btn = gr.Button("Extract Information", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### Results")

                with gr.Tabs():
                    with gr.Tab("Summary"):
                        summary_output = gr.Textbox(
                            label="Extraction Summary",
                            lines=20,
                            max_lines=30,
                            show_copy_button=True
                        )

                    with gr.Tab("Detailed JSON"):
                        json_output = gr.Code(
                            label="Complete Results (JSON)",
                            language="json",
                            lines=20
                        )

        gr.Markdown(
            """
            ---
            ### How to use:
            1. **Upload PDFs**: Click or drag-and-drop one or more PDF files
            2. **Adjust workers**: Set the number of parallel workers (default: 4)
            3. **Extract**: Click the "Extract Information" button
            4. **View results**: Check the Summary tab for an overview or the JSON tab for complete data

            **Note**: Processing time depends on file size and number of files.
            Parallel processing significantly speeds up batch operations.
            """
        )

        # Wire up the event handler
        process_btn.click(
            fn=process_pdfs_parallel,
            inputs=[file_input, max_workers],
            outputs=[summary_output, json_output]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
