import gradio as gr
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
    Returns: (specifications_text, paths_list)
    """
    # Handle empty input
    if not files:
        return "No files uploaded.", []

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

    # Build clean, human-readable output
    lines = []

    for i, result in enumerate(results, 1):
        if i > 1:
            lines.append("\n" + "="*80 + "\n")

        lines.append(f"FILE: {result.get('filename', 'unknown')}")

        if result.get("status") == "success":
            lines.append(f"Pages: {result.get('page_count', 0)} | Diagrams: {len(result.get('diagrams', []))}\n")

            # Show the LLM extracted specifications
            specs = result.get("specifications", "No specifications found")
            # Ensure specs is a string (handle lists or other types)
            if isinstance(specs, list):
                specs = "\n".join(str(s) for s in specs)
            elif not isinstance(specs, str):
                specs = str(specs)
            lines.append(specs)
        else:
            lines.append(f"\nError: {result.get('error', 'Unknown error')}")

    specifications_text = "\n".join(lines)

    # Extract diagram paths
    paths = []
    for result in results:
        for diagram in result.get("diagrams", []):
            path = diagram.get("path")
            if path:
                paths.append(path)

    # Return only simple types
    return specifications_text, paths


def process_files(files, max_workers, llm_provider):
    """
    Gradio wrapper - calls pipeline and returns result.
    NO operations, NO variables, just pass through.
    """
    return run_extraction_pipeline(files, max_workers, llm_provider)


def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(title="PDF Technical Specification & Diagram Extractor") as demo:
        gr.Markdown("# PDF Technical Specification & Diagram Extractor")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload PDF Files",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath"
                )

                parser_type = gr.Dropdown(
                    choices=["Technical Drawing Parser"],
                    value="Technical Drawing Parser",
                    label="Parser Type",
                    interactive=True
                )

                max_workers = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=4,
                    step=1,
                    label="Parallel Workers"
                )

                process_btn = gr.Button("Extract", variant="primary", size="lg")

        with gr.Tabs():
            with gr.Tab("Specifications"):
                summary_output = gr.Textbox(
                    label="Extracted Specifications",
                    lines=25,
                    max_lines=40
                )

            with gr.Tab("Diagrams"):
                diagram_gallery = gr.Gallery(
                    label="Extracted Diagrams",
                    show_label=True,
                    columns=3,
                    height="auto",
                    object_fit="contain"
                )

        # Connect the minimal wrapper (parser_type maps to llm_provider for now)
        process_btn.click(
            fn=process_files,
            inputs=[file_input, max_workers, parser_type],
            outputs=[summary_output, diagram_gallery]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
