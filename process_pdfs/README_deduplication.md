# PDF Processing Pipeline - Deduplication System

This directory contains the deduplication system for the PDF processing pipeline. The system identifies and removes duplicate text chunks from merged JSON files while preserving all chunk IDs and metadata.

## Overview

The deduplication system consists of two main components:

1. **`deduplicate_chunks.py`** - Core deduplication module with functions for analyzing and removing duplicates
2. **`batch_deduplicate.py`** - Batch processing script for running deduplication on multiple papers

## How It Works

### Deduplication Process

1. **Text Analysis**: Chunks are grouped by their text content (ignoring whitespace differences)
2. **Duplicate Detection**: Identifies chunks with identical text content
3. **Metadata Preservation**: Keeps the chunk with the most metadata while preserving all chunk IDs
4. **Output**: Creates a deduplicated file with merged chunk IDs and deduplication statistics

### Chunk ID Handling

- **Original IDs**: All chunk IDs from duplicate chunks are preserved in a `chunk_ids` list
- **Primary ID**: The first chunk ID becomes the primary `chunk_id`
- **Reference**: Original primary ID is stored as `original_chunk_id` for traceability

## Usage

### Individual File Processing

```python
from deduplicate_chunks import process_merged_json_file
from pathlib import Path

# Process a single merged_v2.json file
result = process_merged_json_file(
    json_file_path=Path("papers/001/pages/merged_v2.json"),
    output_path=Path("papers/001/pages/merged_v2_deduplicated.json")  # Optional
)

print(f"Processed: {result['success']}")
print(f"Duplicates removed: {result['deduplication_info']['duplicates_removed']}")
```

### Batch Processing

```bash
# Process all papers in the default METABEEAI_DATA_DIR
python batch_deduplicate.py

# Dry run to analyze without making changes
python batch_deduplicate.py --dry-run

# Process specific paper range
python batch_deduplicate.py --start-paper 1 --end-paper 10

# Process papers in a specific directory
python batch_deduplicate.py --base-dir /path/to/papers

# Save results to specific file
python batch_deduplicate.py --output results.json

# Verbose logging
python batch_deduplicate.py --verbose
```

### Command Line Options

- `--base-dir`: Base directory containing paper folders (defaults to METABEEAI_DATA_DIR from config)
- `--dry-run`: Analyze files without making changes
- `--start-paper`: First paper number to process (inclusive)
- `--end-paper`: Last paper number to process (inclusive)
- `--output`: Output file for results summary
- `--verbose`: Enable verbose logging

## File Structure

The system expects the following directory structure:

```
METABEEAI_DATA_DIR/
├── papers/
│   ├── 001/
│   │   ├── pages/
│   │   │   ├── merged_v2.json          # Input file
│   │   │   └── main_*.pdf.json         # Individual page files (can be removed after merging)
│   │   └── 001_main.pdf
│   ├── 002/
│   │   ├── pages/
│   │   │   └── merged_v2.json
│   │   └── 002_main.pdf
│   └── ...
```

## Output Files

### Deduplicated JSON

The deduplicated `merged_v2.json` file contains:

- **Original chunks**: All unique text chunks
- **Merged chunk IDs**: Lists of all chunk IDs for each unique text
- **Deduplication metadata**: Statistics about the deduplication process

### Results Summary

The batch processing script generates a summary JSON file with:

- Processing statistics
- Per-paper results
- Total duplicates removed
- Error information (if any)

## Integration with Pipeline

### PDF Processing Workflow

1. **Convert PDFs**: Use Vision Agentic Document Analysis to convert PDF pages to JSON
2. **Merge Pages**: Use `merger.py` to combine individual page JSONs into `merged_v2.json`
3. **Deduplicate**: Run the deduplication system to remove duplicates
4. **LLM Pipeline**: The `metabeeai_llm` pipeline now expects deduplicated `merged_v2.json` files

### Configuration

The system automatically uses the `METABEEAI_DATA_DIR` from your `config.py` file. Make sure this is set correctly in your `.env` file:

```bash
METABEEAI_DATA_DIR=/path/to/your/papers/directory
```

## Benefits

1. **Reduced Processing**: Eliminates duplicate text chunks from LLM processing
2. **Preserved Context**: Maintains all chunk IDs for traceability
3. **Improved Quality**: Prevents bias from duplicate information
4. **Efficient Storage**: Reduces file sizes and processing time
5. **Configurable**: Easy to adjust processing parameters and ranges

## Error Handling

The system includes comprehensive error handling:

- **File Not Found**: Gracefully handles missing files
- **JSON Parsing Errors**: Reports and continues with other files
- **Permission Issues**: Logs access problems
- **Partial Failures**: Continues processing other papers if one fails

## Monitoring and Logging

- **Progress Tracking**: Shows progress bars for batch operations
- **Detailed Logging**: Configurable log levels for debugging
- **Results Summary**: Comprehensive output with statistics
- **Error Reporting**: Clear error messages with context

## Example Output

```
============================================================
BATCH DEDUPLICATION SUMMARY
============================================================
Status: completed
Total papers: 25
Processed: 25
Total duplicates removed: 156
Dry run: False
Base directory: /Users/user/Documents/MetaBeeAI_dataset2/papers

Paper 001: 12 duplicates removed (15% reduction)
Paper 002: 8 duplicates removed (10% reduction)
...
```

## Troubleshooting

### Common Issues

1. **No files found**: Check that `METABEEAI_DATA_DIR` is set correctly
2. **Permission errors**: Ensure read/write access to paper directories
3. **JSON parsing errors**: Verify that `merged_v2.json` files are valid JSON
4. **Import errors**: Make sure you're running from the correct directory

### Debug Mode

Use the `--verbose` flag to see detailed logging information:

```bash
python batch_deduplicate.py --verbose --dry-run
```

This will show:
- File discovery process
- Individual file processing details
- Deduplication statistics per file
- Any errors or warnings

## Future Enhancements

- **Parallel Processing**: Process multiple files simultaneously
- **Incremental Updates**: Only process files that have changed
- **Custom Filters**: Allow filtering by chunk type or content
- **Backup Creation**: Automatic backup of original files before deduplication
