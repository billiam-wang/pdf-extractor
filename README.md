# PDF Technical Specification & Diagram Extractor

An AI-powered Gradio web application for extracting technical specifications and diagrams from PDF documents with drag-and-drop functionality and parallel processing capabilities.

## Features

- **AI-Powered Specification Extraction**: Automatically identifies and extracts technical specifications including:
  - Materials and composition
  - Electrical properties (voltage, current, resistance, capacitance, etc.)
  - Physical dimensions and measurements
  - Performance characteristics
  - Operating conditions (temperature, pressure, etc.)
  - Standards and certifications
  - Part numbers and model information

- **Diagram Extraction**: Automatically extracts all images and diagrams from PDFs as individual image files
- **Drag-and-Drop Interface**: Easy file upload with support for multiple PDFs
- **Parallel Processing**: Process multiple files simultaneously for faster results
- **Multi-LLM Support**: Choose from Anthropic Claude, OpenAI GPT-4, or Databricks models
- **Visual Gallery**: View all extracted diagrams in an organized gallery
- **Databricks App Ready**: Configured for deployment as a Databricks App

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-extractor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your LLM API credentials (choose one):

**For Anthropic (Claude):**
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export LLM_PROVIDER="anthropic"
```

**For OpenAI (GPT-4):**
```bash
export OPENAI_API_KEY="your-openai-api-key"
export LLM_PROVIDER="openai"
```

**For Databricks:**
```bash
export DATABRICKS_API_URL="https://your-workspace.cloud.databricks.com/serving-endpoints/your-endpoint/invocations"
export DATABRICKS_TOKEN="your-databricks-token"
export LLM_PROVIDER="databricks"
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:7860
```

## Usage

1. **Configure LLM Provider**: Select your preferred AI provider from the dropdown (Anthropic, OpenAI, or Databricks)
2. **Upload PDFs**: Click the upload area or drag-and-drop one or more technical PDF files
3. **Adjust Workers**: Set the number of parallel workers (1-10, default: 4)
4. **Extract**: Click the "Extract Specifications & Diagrams" button
5. **View Results**:
   - **Specifications Summary Tab**: AI-extracted technical specifications and extraction summary
   - **Extracted Diagrams Tab**: Visual gallery of all diagrams found in the PDFs
   - **Detailed JSON Tab**: Complete extraction results in structured JSON format

## Databricks App Deployment

This application is configured for deployment as a Databricks App using the `app.yaml` configuration file.

### Deployment Steps

1. **Prepare your Databricks workspace**:
   - Ensure you have appropriate permissions to create apps
   - Have a Databricks workspace with Apps feature enabled

2. **Configure environment variables** in your Databricks App settings:
   ```bash
   ANTHROPIC_API_KEY=your-api-key
   # OR
   OPENAI_API_KEY=your-api-key
   # OR
   DATABRICKS_API_URL=your-model-endpoint
   DATABRICKS_TOKEN=your-token

   LLM_PROVIDER=anthropic  # or openai or databricks
   ```

3. **Deploy the app**:
   - Upload the repository to your Databricks workspace
   - Use the Databricks Apps UI or CLI to deploy
   - The app will automatically use the `app.yaml` configuration

### app.yaml Configuration

```yaml
command: ["python", "app.py"]
env:
  - name: GRADIO_SERVER_NAME
    value: "0.0.0.0"
  - name: GRADIO_SERVER_PORT
    value: "7860"
```

## Architecture

### Parallel Processing

The app uses Python's `ThreadPoolExecutor` to process multiple PDF files concurrently:

- Default: 4 parallel workers
- Configurable: 1-10 workers via the UI
- Each file is processed independently
- Results are collected as processing completes

### PDF Processing Pipeline

For each PDF file, the app performs:

1. **Text Extraction**: Extracts all text content using PyMuPDF (fitz)
2. **Specification Analysis**: Sends text to LLM for intelligent specification extraction
3. **Image Extraction**: Identifies and extracts all images/diagrams
4. **Result Compilation**: Combines specifications and diagrams into structured output

### LLM Integration

The app supports three LLM providers:

- **Anthropic (Claude 3.5 Sonnet)**: Best for technical document analysis
- **OpenAI (GPT-4o)**: Strong general-purpose extraction
- **Databricks**: Use your own model serving endpoints

### Key Functions

- `extract_specifications_with_llm(text, provider)`: Analyzes text and extracts specifications using LLM
- `extract_images_from_pdf(file_path)`: Extracts all images/diagrams as separate files
- `extract_pdf_info(file_path, llm_provider)`: Orchestrates extraction for a single PDF
- `process_pdfs_parallel(files, max_workers, llm_provider)`: Handles parallel batch processing
- `create_interface()`: Builds the Gradio UI

## File Structure

```
pdf-extractor/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── app.yaml              # Databricks App configuration
├── README.md             # This file
├── .gitignore            # Git exclusions
└── extracted_images/     # Output directory for extracted diagrams (created at runtime)
```

## Requirements

- Python 3.8+
- gradio >= 4.0.0
- PyMuPDF >= 1.23.0
- Pillow >= 10.0.0
- anthropic >= 0.18.0 (for Claude)
- openai >= 1.12.0 (for GPT-4)
- requests >= 2.31.0 (for Databricks)

## Error Handling

The application includes robust error handling:

- Individual file errors don't stop batch processing
- LLM errors are caught and reported
- Image extraction failures are logged
- Detailed error messages in results
- Tracebacks available in JSON output for debugging

## Performance Tips

- **Small batches**: For very large PDFs, reduce the number of parallel workers
- **Large batches**: For many small PDFs, increase parallel workers (up to 10)
- **Memory**: Monitor memory usage when processing many large files simultaneously
- **LLM costs**: Be aware that each PDF processes text through the LLM API

## Output

### Specifications Summary

The summary tab displays:
- Processing statistics (success/error counts)
- Total diagrams extracted
- Per-file technical specifications extracted by the LLM
- List of extracted diagrams with metadata (page, dimensions, file size)

### Extracted Diagrams

All diagrams are:
- Saved to `extracted_images/` directory
- Named with pattern: `{filename}_page{N}_img{M}.{ext}`
- Displayed in a visual gallery
- Available for download

### JSON Output

Complete structured data including:
- Filename and processing status
- Page count
- Full specifications text
- Diagram metadata (paths, dimensions, formats)
- Error details (if any)

## Limitations

- Supports only PDF files
- Text extraction quality depends on PDF structure
- OCR is not included (scanned PDFs without embedded text may not extract well)
- LLM context limits to first 10,000 characters of text
- Requires API keys for LLM providers

## Future Enhancements

Potential improvements:
- OCR support for scanned PDFs using pytesseract
- Vision model support for extracting specs directly from diagram images
- Export specifications to structured formats (CSV, Excel)
- Batch download of all extracted diagrams as ZIP
- Document classification and categorization
- Named entity recognition for part numbers
- Integration with vector databases for semantic search

## Troubleshooting

### "No API key set" error
Make sure you've set the appropriate environment variable:
- `ANTHROPIC_API_KEY` for Claude
- `OPENAI_API_KEY` for GPT-4
- `DATABRICKS_API_URL` and `DATABRICKS_TOKEN` for Databricks

### No diagrams extracted
Some PDFs embed images in ways that are difficult to extract. The app extracts images stored as objects in the PDF. Vector graphics and embedded diagrams may not be extracted.

### LLM errors
Check your API key is valid and you have sufficient credits/quota. Network connectivity issues can also cause failures.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

[Add support contact information here]

## Credits

- Built with [Gradio](https://gradio.app/)
- PDF processing with [PyMuPDF](https://pymupdf.readthedocs.io/)
- AI extraction powered by Anthropic Claude, OpenAI GPT-4, or Databricks
