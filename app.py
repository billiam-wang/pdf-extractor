import gradio as gr
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import json
import os

from parser import TechnicalDrawingParser


def process_pdfs_parallel(files, max_workers: int = 4, llm_provider: Optional[str] = None):
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

    # Initialize parser (technical drawing parser is the default)
    parser = TechnicalDrawingParser(llm_provider=llm_provider)

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(parser.parse, file): file for file in files}

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

    # Get default LLM provider from environment
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
