import gradio as gr
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os

from parser import TechnicalDrawingParser


def parse_single_pdf(file_path, llm_provider):
    """Parse a single PDF file."""
    parser = TechnicalDrawingParser(llm_provider=llm_provider)
    return parser.parse(file_path)


def format_results(results):
    """Format results into summary text and JSON."""
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

    return summary_text, detailed_json


def process_files(files, max_workers, llm_provider):
    """Process multiple PDF files and return results."""
    if not files:
        return "No files uploaded.", "{}", []

    results = []
    all_diagrams = []

    # Process files in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(parse_single_pdf, file, llm_provider): file
            for file in files
        }

        # Collect results as they complete
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                result = future.result()
                results.append(result)

                # Collect diagram paths
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

    # Format results
    summary_text, detailed_json = format_results(results)

    return summary_text, detailed_json, all_diagrams


def create_interface():
    """Create the Gradio interface."""

    default_llm_provider = os.getenv("LLM_PROVIDER", "anthropic")

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
                    value=default_llm_provider,
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
            1. **Configure LLM**: Set your API key as environment variable
            2. **Upload PDFs**: Click or drag-and-drop one or more technical PDF files
            3. **Select Provider**: Choose your preferred AI model provider
            4. **Adjust Workers**: Set parallel processing level (default: 4)
            5. **Extract**: Click the extract button and wait for processing
            6. **Review Results**: Check the three tabs for different views of the data

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

        # Wire up the event handler - single simple function call
        process_btn.click(
            fn=process_files,
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
