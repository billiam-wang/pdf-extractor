# PDF Information Extractor

A simple Gradio-based web application for extracting text content and metadata from PDF files with drag-and-drop functionality and parallel processing capabilities.

## Features

- **Drag-and-Drop Interface**: Easy file upload with support for multiple PDFs
- **Parallel Processing**: Process multiple files simultaneously for faster results
- **Text Extraction**: Extract all text content from PDF pages
- **Metadata Extraction**: Retrieve document metadata (title, author, creation date, etc.)
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

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:7860
```

## Usage

1. **Upload PDFs**: Click the upload area or drag-and-drop one or more PDF files
2. **Adjust Workers**: Set the number of parallel workers (1-10, default: 4)
3. **Extract Information**: Click the "Extract Information" button
4. **View Results**:
   - **Summary Tab**: Overview with page counts, metadata, and text previews
   - **JSON Tab**: Complete extraction results in JSON format

## Databricks App Deployment

This application is configured for deployment as a Databricks App using the `app.yaml` configuration file.

### Deployment Steps

1. **Prepare your Databricks workspace**:
   - Ensure you have appropriate permissions to create apps
   - Have a Databricks workspace with Apps feature enabled

2. **Deploy the app**:
   - Upload the repository to your Databricks workspace
   - Use the Databricks Apps UI or CLI to deploy
   - The app will automatically use the `app.yaml` configuration

3. **Configuration**:
   - The app runs on port 7860 by default
   - All dependencies are specified in `requirements.txt`
   - Environment variables can be modified in `app.yaml`

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

### PDF Processing

The app combines two libraries for comprehensive extraction:

- **pdfplumber**: Primary text extraction with page-by-page processing
- **PyPDF2**: Metadata extraction (title, author, dates, etc.)

### Key Functions

- `extract_pdf_info(file_path)`: Extracts text and metadata from a single PDF
- `process_pdfs_parallel(files, max_workers)`: Orchestrates parallel processing
- `create_interface()`: Builds the Gradio UI

## File Structure

```
pdf-extractor/
├── app.py              # Main Gradio application
├── requirements.txt    # Python dependencies
├── app.yaml           # Databricks App configuration
└── README.md          # This file
```

## Requirements

- Python 3.8+
- gradio >= 4.0.0
- PyPDF2 >= 3.0.0
- pdfplumber >= 0.10.0
- Pillow >= 10.0.0

## Error Handling

The application includes robust error handling:

- Individual file errors don't stop batch processing
- Detailed error messages in results
- Tracebacks available in JSON output for debugging

## Performance Tips

- **Small batches**: For very large PDFs, reduce the number of parallel workers
- **Large batches**: For many small PDFs, increase parallel workers (up to 10)
- **Memory**: Monitor memory usage when processing many large files simultaneously

## Limitations

- Supports only PDF files
- Text extraction quality depends on PDF structure (scanned PDFs may not extract well)
- OCR is not included (consider adding pytesseract for scanned documents)

## Future Enhancements

Potential improvements:
- OCR support for scanned PDFs
- Additional export formats (CSV, Excel)
- Document classification
- Named entity recognition
- Summary generation using LLMs

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

[Add support contact information here]
