import gradio as gr
import os
import markdown

from parser import TechnicalDrawingParser
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_single_pdf(file_path, llm_provider):
    parser = TechnicalDrawingParser(llm_provider=llm_provider)
    return parser.parse(file_path)


def run_extraction_pipeline(files, max_workers, llm_provider):
    # Handle empty input
    if not files:
        return "<p>No files uploaded.</p>", []

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

    # Build HTML output with collapsible sections for each document
    html_parts = []
    html_parts.append('''
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f5f5f5;
            font-weight: 600;
        }
        tr:nth-child(even) {
            background-color: #fafafa;
        }
        tr:hover {
            background-color: #f0f0f0;
        }
    </style>
    <div style="font-family: system-ui, -apple-system, sans-serif;">
    ''')

    for i, result in enumerate(results):
        filename = result.get('filename', 'unknown')
        status = result.get('status', 'unknown')

        # Create collapsible section for each document
        html_parts.append(f'''
        <details open style="margin-bottom: 20px; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background: #f9f9f9;">
            <summary style="cursor: pointer; font-size: 18px; font-weight: 600; margin-bottom: 12px; color: #333;">
                📄 {filename}
                <span style="font-size: 14px; font-weight: normal; color: #666; margin-left: 8px;">
                    ({result.get('page_count', 0)} pages | {len(result.get('diagrams', []))} diagrams)
                </span>
            </summary>
            <div style="margin-top: 12px; padding: 12px; background: white; border-radius: 4px;">
        ''')

        if status == "success":
            # Show specifications
            specs = result.get("specifications", "No specifications found")

            # Try to parse JSON response
            try:
                import json
                if isinstance(specs, str):
                    spec_data = json.loads(specs)
                else:
                    spec_data = specs

                # Build HTML for structured JSON display
                html_parts.append('<div style="line-height: 1.6; color: #333;">')

                # Display ID
                if 'id' in spec_data:
                    html_parts.append(f'<p><strong>ID:</strong> {spec_data.get("id", "N/A")}</p>')

                # Display specifications
                if 'specifications' in spec_data:
                    spec_details = spec_data['specifications']
                    html_parts.append('<h4 style="margin-top: 16px; margin-bottom: 8px;">Specifications</h4>')

                    # Electrical properties
                    if 'electrical' in spec_details and spec_details['electrical']:
                        html_parts.append('<p><strong>Electrical:</strong></p>')
                        html_parts.append('<ul>')
                        for item in spec_details['electrical']:
                            html_parts.append(f'<li>{item}</li>')
                        html_parts.append('</ul>')

                    # Material properties
                    if 'material' in spec_details and spec_details['material']:
                        html_parts.append('<p><strong>Material:</strong></p>')
                        html_parts.append('<ul>')
                        for item in spec_details['material']:
                            html_parts.append(f'<li>{item}</li>')
                        html_parts.append('</ul>')

                    # Operating temperature
                    if 'operation_temperature' in spec_details and spec_details['operation_temperature']:
                        html_parts.append(f'<p><strong>Operating Temperature:</strong> {spec_details["operation_temperature"]}</p>')

                html_parts.append('</div>')

            except (json.JSONDecodeError, TypeError, KeyError) as e:
                # Fallback: display as plain text
                if isinstance(specs, str):
                    html_parts.append(f'<div style="line-height: 1.6; color: #333;"><pre>{specs}</pre></div>')
                else:
                    html_parts.append(f'<div style="line-height: 1.6; color: #333;">{str(specs)}</div>')

            # Show diagram info if any
            diagrams = result.get("diagrams", [])
            if diagrams:
                html_parts.append('<div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid #eee;">')
                html_parts.append('<strong>Extracted Diagrams:</strong>')
                html_parts.append('<ul style="margin-top: 8px;">')
                for diag in diagrams:
                    html_parts.append(
                        f'<li>{diag.get("filename", "unknown")} - '
                        f'Page {diag.get("page", "?")}, '
                        f'{diag.get("width", 0)}×{diag.get("height", 0)}px</li>'
                    )
                html_parts.append('</ul></div>')
        else:
            html_parts.append(f'<div style="color: #d32f2f;">❌ Error: {result.get("error", "Unknown error")}</div>')

        html_parts.append('</div></details>')

    html_parts.append('</div>')
    html_output = ''.join(html_parts)

    # Extract diagram paths
    paths = []
    for result in results:
        for diagram in result.get("diagrams", []):
            path = diagram.get("path")
            if path:
                paths.append(path)

    # Return only simple types
    return html_output, paths


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
                summary_output = gr.HTML(
                    label="Extracted Specifications"
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
