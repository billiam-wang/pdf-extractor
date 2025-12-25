import gradio as gr
import json
import os

from parser import TechnicalDrawingParser
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_single_pdf(file_path, llm_provider):
    """Parse a single PDF file."""
    parser = TechnicalDrawingParser(llm_provider=llm_provider)
    return parser.parse(file_path)


def run_extraction_pipeline(files, max_workers, llm_provider):
    """
    Complete extraction pipeline that returns only simple types.
    All dict operations happen inside this function.
    Returns: (summary_str, json_str, paths_list)
    """
    # Handle empty input
    if not files:
        return "No files uploaded.", "{}", []

    # Process all files in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(parse_single_pdf, file, llm_provider): file
            for file in files
        }

        for future in as_completed(futures):
            file_path = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    "filename": os.path.basename(file_path),
                    "status": "error",
                    "error": str(e),
                    "specifications": "",
                    "diagrams": [],
                    "page_count": 0
                })

    # Build summary text
    lines = []
    success = sum(1 for r in results if r.get("status") == "success")
    errors = len(results) - success
    total_diagrams = sum(len(r.get("diagrams", [])) for r in results)

    lines.append(f"Processed {len(results)} file(s):\n")
    lines.append(f"✅ Successful: {success}")
    lines.append(f"❌ Errors: {errors}")
    lines.append(f"🖼️  Total diagrams: {total_diagrams}\n")

    for i, result in enumerate(results, 1):
        lines.append(f"\n{'='*80}")
        lines.append(f"File {i}: {result.get('filename', 'unknown')}")
        lines.append(f"Status: {result.get('status', 'unknown').upper()}")

        if result.get("status") == "success":
            lines.append(f"Pages: {result.get('page_count', 0)}")
            lines.append(f"Diagrams extracted: {len(result.get('diagrams', []))}")

            lines.append("\n--- TECHNICAL SPECIFICATIONS ---")
            lines.append(result.get("specifications", "No specifications found"))

            diagrams = result.get("diagrams", [])
            if diagrams:
                lines.append("\n--- EXTRACTED DIAGRAMS ---")
                for diag in diagrams:
                    lines.append(
                        f"  • {diag.get('filename', 'unknown')} - "
                        f"Page {diag.get('page', '?')}, "
                        f"{diag.get('width', 0)}x{diag.get('height', 0)}px, "
                        f"{diag.get('size_bytes', 0) / 1024:.1f} KB"
                    )
        else:
            lines.append(f"Error: {result.get('error', 'Unknown error')}")

    summary_text = "\n".join(lines)

    # Convert to JSON string
    json_text = json.dumps(results, indent=2, ensure_ascii=False, default=str)

    # Extract diagram paths
    paths = []
    for result in results:
        for diagram in result.get("diagrams", []):
            path = diagram.get("path")
            if path:
                paths.append(path)

    # Return only simple types
    return summary_text, json_text, paths


def process_files(files, max_workers, llm_provider):
    """
    Gradio wrapper - calls pipeline and returns result.
    NO operations, NO variables, just pass through.
    """
    return run_extraction_pipeline(files, max_workers, llm_provider)


def create_interface():
    """Create the Gradio interface."""

    default_llm_provider = os.getenv("LLM_PROVIDER", "anthropic")

    with gr.Blocks(title="PDF Technical Specification & Diagram Extractor") as demo:
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
                    max_lines=40
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

        # Connect the minimal wrapper
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
        share=False,
        theme=gr.themes.Soft()
    )
