# PDF Processing Pipeline

This folder contains scripts for converting PDF scientific papers into structured JSON format suitable for LLM-based information extraction.

## Overview

The PDF processing pipeline converts raw PDF papers into structured JSON chunks through four main steps:

1. **Split PDFs** - Break papers into overlapping 2-page segments
2. **Vision API Processing** - Extract text and structure using Landing AI's Vision Agentic API
3. **Merge JSON Files** - Combine individual page JSON files into a single document
4. **Deduplicate Chunks** - Remove duplicate text chunks from overlapping pages

---

## Quick Start

### Prerequisites

1. **Environment Setup**:
```bash
# Activate your virtual environment
source ../venv/bin/activate  # On Mac/Linux
# Or: ..\venv\Scripts\activate  # On Windows
```

2. **API Keys**: Configure your API keys in the `.env` file:
```bash
# Copy the example environment file
cp ../env.example ../.env

# Edit the .env file and add your API key
# LANDING_AI_API_KEY=your_landing_ai_api_key_here
```

The `.env` file is located in the project root directory and is hidden from git for security.

3. **Input Data Format**: Your papers must be organized as follows:
```
papers/
├── 001/
│   └── 001_main.pdf
├── 002/
│   └── 002_main.pdf
├── 003/
│   └── 003_main.pdf
...
```

Each paper should have:
- A folder with a 3-digit numeric name (e.g., `001`, `002`, `003`)
- A PDF file named `{folder_number}_main.pdf` inside that folder

---

### Basic Usage - Complete Pipeline

Run all steps for all papers in directory:
```bash
python process_all.py
```

Run all steps for papers 1-10:
```bash
python process_all.py --start 1 --end 10
```

Run with a custom directory:
```bash
python process_all.py --dir /path/to/papers --start 1 --end 10
```

**Merge-only mode** (skip expensive PDF splitting and API processing):
```bash
# Process all papers - only merge and deduplicate
python process_all.py --merge-only

# Process specific papers - only merge and deduplicate
python process_all.py --merge-only --start 1 --end 10
```

---

## Core Files

### 1. `process_all.py` - **Main Pipeline Runner**

**Purpose**: Orchestrates all four steps of the PDF processing pipeline in sequence.

**Usage**:
```bash
# Process all papers (all steps)
python process_all.py

# Process papers 1-10 (all steps)
python process_all.py --start 1 --end 10

# Process papers 50-100
python process_all.py --start 50 --end 100

# Merge-only mode (skip expensive PDF splitting and API processing)
python process_all.py --merge-only

# Merge-only for specific papers
python process_all.py --merge-only --start 1 --end 10

# Filter out marginalia chunks during merging
python process_all.py --start 1 --end 10 --filter-chunk-type marginalia
```

**Command-line options**:
- `--start N`: First paper number to process (optional; defaults to first paper in directory)
- `--end N`: Last paper number to process (optional; defaults to last paper in directory)
- `--dir PATH`: Custom papers directory (default: from config/env)
- `--merge-only`: Only run merge and deduplication steps (skip expensive PDF splitting and API processing)
- `--skip-split`: Skip PDF splitting step
- `--skip-api`: Skip Vision API processing step
- `--skip-merge`: Skip JSON merging step
- `--skip-deduplicate`: Skip chunk deduplication step
- `--filter-chunk-type TYPE [TYPE ...]`: Filter out specific chunk types (e.g., marginalia, figure)

**Output**: Creates the following files for each paper:
- `papers/XXX/pages/main_p01-02.pdf`, `main_p02-03.pdf`, etc. (split PDFs)
- `papers/XXX/pages/main_p01-02.pdf.json`, etc. (API responses)
- `papers/XXX/pages/merged_v2.json` (final merged and deduplicated file)

---

### 2. `split_pdf.py` - PDF Splitter

**Purpose**: Splits multi-page PDFs into overlapping 2-page segments to help the Vision API maintain context across page boundaries.

**Why overlapping pages?**: Scientific papers often have content that spans across pages (tables, paragraphs). Overlapping ensures we don't lose information at page boundaries.

**Usage as standalone**:
```bash
python split_pdf.py /path/to/papers
```

**How it works**:
1. Finds all `XXX_main.pdf` files in paper folders
2. Creates a `pages/` subdirectory in each paper folder
3. Generates overlapping 2-page PDFs:
   - `main_p01-02.pdf` (pages 1-2)
   - `main_p02-03.pdf` (pages 2-3)
   - `main_p03-04.pdf` (pages 3-4)
   - etc.

**Example**:
A 10-page paper will generate 9 split PDFs with overlapping pages.

---

### 3. `va_process_papers.py` - Vision API Processor

**Purpose**: Processes each split PDF through Landing AI's Vision Agentic Document Analysis API to extract text and structure.

**Usage as standalone**:
```bash
# Process all papers
python va_process_papers.py --dir data/papers

# Start from a specific paper (useful for resuming)
python va_process_papers.py --dir data/papers --start 059
```

**Command-line options**:
- `--dir PATH`: Papers directory (default: data/papers)
- `--start XXX`: Starting folder number (e.g., 059)

**How it works**:
1. Finds all split PDF files in `papers/XXX/pages/`
2. Sends each PDF to the Vision Agentic API
3. Saves the JSON response as `{pdf_filename}.json`
4. Skips files that already have JSON outputs (resume-friendly)
5. Logs all processing activity with timestamps

**Output**: Creates JSON files with this structure:
```json
{
  "data": {
    "chunks": [
      {
        "chunk_id": "unique_id",
        "text": "Extracted text content...",
        "chunk_type": "paragraph",
        "grounding": [
          {
            "page": 0,
            "bbox": [x1, y1, x2, y2]
          }
        ],
        "metadata": {...}
      }
    ]
  }
}
```

**Important**: This step requires a valid `LANDING_AI_API_KEY` in your `.env` file.

---

### 4. `merger.py` - JSON Merger

**Purpose**: Combines individual page JSON files into a single `merged_v2.json` file per paper, handling overlapping pages correctly.

**Usage as standalone**:
```bash
# Merge all papers
python merger.py --basepath /path/to/data

# Filter out marginalia chunks
python merger.py --basepath /path/to/data --filter-chunk-type marginalia

# Filter multiple chunk types
python merger.py --basepath /path/to/data --filter-chunk-type marginalia figure
```

**Command-line options**:
- `--basepath PATH`: Base path containing the `papers/` folder
- `--filter-chunk-type TYPE [TYPE ...]`: Chunk types to exclude from output

**How it works**:
1. Finds all `main_*.json` files in `papers/XXX/pages/`
2. Adjusts page numbers to account for overlapping pages
3. Merges all chunks into a single JSON structure
4. Optionally filters out specified chunk types
5. Saves as `merged_v2.json`

**Page number adjustment**: Since pages overlap, the merger maps overlapping pages to the same global page number to avoid duplication.

**Example**:
- File 1 (pages 1-2): Global pages 1-2
- File 2 (pages 2-3): Page 2 maps to global page 2, page 3 becomes global page 3
- File 3 (pages 3-4): Page 3 maps to global page 3, page 4 becomes global page 4

**Output format**:
```json
{
  "data": {
    "chunks": [
      {
        "chunk_id": "unique_id",
        "text": "...",
        "chunk_type": "paragraph",
        "grounding": [{"page": 0, "bbox": [...]}],
        "metadata": {...}
      }
    ]
  }
}
```

---

### 5. `deduplicate_chunks.py` - Chunk Deduplication Module

**Purpose**: Provides functions to identify and remove duplicate text chunks that result from overlapping pages.

**Usage as a module**:
```python
from deduplicate_chunks import analyze_chunk_uniqueness, deduplicate_chunks

# Analyze duplication
analysis = analyze_chunk_uniqueness(chunks)
print(f"Found {analysis['duplicate_chunks']} duplicates")

# Deduplicate
deduplicated_chunks = deduplicate_chunks(chunks)
```

**Key functions**:
- `analyze_chunk_uniqueness(chunks)` - Returns statistics about duplicates
- `deduplicate_chunks(chunks)` - Removes duplicates while preserving all chunk IDs
- `get_duplicate_summary(chunks)` - Human-readable summary of duplicates
- `process_merged_json_file(path)` - Process a single merged JSON file

**How deduplication works**:
1. Groups chunks by text content
2. For duplicate groups, keeps one chunk but preserves all chunk IDs
3. Adds metadata about the deduplication process

**Duplicate handling**: When duplicates are found, the deduplicated chunk includes:
- `chunk_id`: Primary chunk ID (first occurrence)
- `chunk_ids`: List of all chunk IDs with the same text
- `original_chunk_id`: Reference to the original ID

---

### 6. `batch_deduplicate.py` - Batch Deduplication Runner

**Purpose**: Processes all `merged_v2.json` files in a directory to remove duplicates.

**Usage as standalone**:
```bash
# Deduplicate all papers
python batch_deduplicate.py

# Deduplicate papers in a range
python batch_deduplicate.py --start-paper 1 --end-paper 10

# Dry run (analyze without modifying files)
python batch_deduplicate.py --dry-run

# Custom directory
python batch_deduplicate.py --base-dir /path/to/papers

# Verbose output
python batch_deduplicate.py --verbose
```

**Command-line options**:
- `--base-dir PATH`: Base directory containing paper folders
- `--start-paper N`: First paper number to process
- `--end-paper N`: Last paper number to process
- `--dry-run`: Analyze files without making changes
- `--output FILE`: Save results summary to file
- `--verbose`, `-v`: Enable verbose logging

**How it works**:
1. Finds all paper folders with `merged_v2.json` files
2. Analyzes each file for duplicate chunks
3. Deduplicates and overwrites the file (unless `--dry-run`)
4. Generates a summary report

**Output**: Creates a summary JSON file with:
```json
{
  "status": "completed",
  "total_papers": 10,
  "processed_papers": 10,
  "total_duplicates_removed": 145,
  "results": [...]
}
```

---

## Individual Script Usage

### Running Steps Separately

If you need to run individual steps (useful for debugging or resuming):

#### Step 1: Split PDFs
```bash
python split_pdf.py /path/to/papers
```

#### Step 2: Process with Vision API
```bash
python va_process_papers.py --dir /path/to/papers
```

#### Step 3: Merge JSON files
```bash
python merger.py --basepath /path/to/data
```

#### Step 4: Deduplicate chunks
```bash
python batch_deduplicate.py --base-dir /path/to/papers
```

---

## Input Data Format

### Required Directory Structure

```
papers/
├── 001/
│   └── 001_main.pdf
├── 002/
│   └── 002_main.pdf
├── 003/
│   └── 003_main.pdf
...
```

**Requirements**:
- Each paper must be in a folder with a 3-digit numeric name
- PDF file must be named `{folder_number}_main.pdf`
- PDFs should be complete scientific papers (not split or partial)

### PDF Requirements

- **Format**: Valid PDF files
- **Content**: Text-based PDFs work best (scanned PDFs may have lower quality)
- **Size**: No strict limits, but very large files may take longer to process
- **Pages**: Multi-page documents are fully supported

---

## Output Data Format

### Final Output: `merged_v2.json`

The pipeline produces a `merged_v2.json` file for each paper with the following structure:

```json
{
  "data": {
    "chunks": [
      {
        "chunk_id": "unique_chunk_identifier",
        "text": "The extracted text content from the PDF...",
        "chunk_type": "paragraph",
        "grounding": [
          {
            "page": 0,
            "bbox": [x1, y1, x2, y2]
          }
        ],
        "chunk_ids": ["id1", "id2"],
        "metadata": {
          "confidence": 0.95,
          "font_size": 12,
          ...
        }
      }
    ]
  },
  "deduplication_info": {
    "original_chunks": 500,
    "unique_chunks": 450,
    "duplicates_removed": 50,
    "duplication_rate": 10.0,
    "duplicate_groups": 25
  }
}
```

### Field Descriptions:

- **chunk_id**: Unique identifier for this chunk
- **text**: Extracted text content
- **chunk_type**: Type of content (paragraph, heading, table, figure, marginalia, etc.)
- **grounding**: Location information
  - **page**: Page number (0-indexed)
  - **bbox**: Bounding box coordinates [x1, y1, x2, y2]
- **chunk_ids**: List of all chunk IDs with identical text (after deduplication)
- **metadata**: Additional information from the Vision API
- **deduplication_info**: Statistics about the deduplication process

This format is designed to be consumed by the LLM pipeline in `../metabeeai_llm/`.

---

## Understanding the Process Flow

### Complete Pipeline Flow

```
Raw PDF → Split PDF → Vision API → Individual JSONs → Merged JSON → Deduplicated JSON
```

**Detailed steps**:

1. **Input**: `001_main.pdf` (10 pages)

2. **After Splitting**:
   ```
   pages/main_p01-02.pdf
   pages/main_p02-03.pdf
   pages/main_p03-04.pdf
   ...
   pages/main_p09-10.pdf
   ```

3. **After API Processing**:
   ```
   pages/main_p01-02.pdf.json
   pages/main_p02-03.pdf.json
   ...
   pages/main_p09-10.pdf.json
   ```

4. **After Merging**:
   ```
   pages/merged_v2.json (contains all chunks with adjusted page numbers)
   ```

5. **After Deduplication**:
   ```
   pages/merged_v2.json (duplicates removed, chunk IDs preserved)
   ```

---

## Troubleshooting

### "LANDING_AI_API_KEY not found"
- **Cause**: API key not configured in `.env` file
- **Fix**: 
  ```bash
  cp ../env.example ../.env
  # Edit .env and add your LANDING_AI_API_KEY
  ```

### "PDF file not found"
- **Cause**: PDF file not named correctly or in wrong location
- **Fix**: Ensure PDFs are named `{folder_number}_main.pdf` and in the correct folder

### "No merged_v2.json files found"
- **Cause**: Merger step hasn't been run yet or failed
- **Fix**: Run `python merger.py --basepath /path/to/data` first

### API processing is slow
- **Cause**: Vision API processes each page individually
- **Solution**: This is normal. Processing time depends on:
  - Number of papers
  - Pages per paper
  - API response time
  - The script will automatically resume if interrupted

### Duplicate chunks remain after deduplication
- **Cause**: Chunks might have slight text differences
- **Fix**: Check the deduplication_info in merged_v2.json for statistics
- **Note**: Only exact text matches are considered duplicates

### Out of API quota
- **Cause**: Too many API calls
- **Fix**: 
  - The script automatically skips already-processed files
  - Use `--start` parameter to resume from a specific paper
  - Contact Landing AI to increase your quota

---

## Advanced Usage

### Merge-Only Mode (Cost-Effective)

If you've already run the expensive PDF splitting and Vision API processing steps, you can use `--merge-only` to only run the merge and deduplication steps:

```bash
# Process all papers - merge and deduplicate only
python process_all.py --merge-only

# Process specific papers - merge and deduplicate only
python process_all.py --merge-only --start 1 --end 10
```

This is useful when:
- You've already processed PDFs through the Vision API
- You want to re-run merging with different filter options
- You want to re-deduplicate after manual edits to JSON files
- You're testing the merge/deduplication logic without API costs

**Note**: Merge-only mode validates that JSON files exist (not PDFs) and automatically skips the split and API steps.

### Processing All Papers Automatically

If you don't specify `--start` and `--end`, the pipeline will automatically detect and process all numeric folders in your papers directory:

```bash
# Process all papers found in the directory
python process_all.py

# Process all papers with merge-only
python process_all.py --merge-only
```

The script will:
1. Scan the papers directory for numeric folders (001, 002, 003, etc.)
2. Sort them numerically
3. Process from the first to the last paper found

### Filtering Chunk Types

You can filter out specific chunk types during merging:

```bash
# Remove marginalia (page numbers, headers, footers)
python process_all.py --start 1 --end 10 --filter-chunk-type marginalia

# Remove multiple types
python process_all.py --start 1 --end 10 --filter-chunk-type marginalia figure

# When running merger separately
python merger.py --basepath /path/to/data --filter-chunk-type marginalia
```

Common chunk types to filter:
- `marginalia` - Headers, footers, page numbers
- `figure` - Figure captions (if you only want main text)
- `table` - Table content (if you only want prose)

### Resuming Processing

If processing is interrupted, the pipeline is resume-friendly:

```bash
# API processing automatically skips existing JSON files
python va_process_papers.py --dir /path/to/papers --start 059

# Deduplication can be re-run on specific papers
python batch_deduplicate.py --start-paper 50 --end-paper 100
```

### Dry Run Mode

Test the pipeline without making changes:

```bash
# Analyze duplication without modifying files
python batch_deduplicate.py --dry-run

# See what would happen
python batch_deduplicate.py --dry-run --verbose
```

---

## Performance Tips

1. **Parallel Processing**: The Vision API processes one file at a time. For faster processing, consider running multiple instances on different paper ranges:
   ```bash
   # Terminal 1
   python process_all.py --start 1 --end 50
   
   # Terminal 2
   python process_all.py --start 51 --end 100
   ```

2. **Resume from Failures**: If processing fails partway through, use `--skip-split` and `--start` to resume:
   ```bash
   python process_all.py --start 75 --end 100 --skip-split
   ```

3. **Monitor Progress**: Check log files created in the papers directory:
   ```bash
   tail -f papers/processing_log_*.txt
   ```

---

## Dependencies

Core dependencies (install via `pip install -r ../requirements.txt`):
- `PyPDF2` - PDF manipulation
- `requests` - API calls
- `python-dotenv` - Environment variable management
- `termcolor` - Colored console output
- `pathlib` - Path operations

---

## Next Steps

After processing your PDFs:

1. Verify output files:
   ```bash
   ls papers/001/pages/merged_v2.json
   ```

2. Check deduplication statistics in the JSON file

3. Proceed to the LLM pipeline:
   ```bash
   cd ../metabeeai_llm
   python llm_pipeline.py --start 1 --end 10
   ```

---

## Related Documentation

- **LLM Pipeline**: See `../metabeeai_llm/README.md` for extracting information from processed papers
- **Data Analysis**: See `../query_database/` for analyzing extracted data
- **Configuration**: See `../config.py` for centralized configuration

---

**Last Updated**: October 2025

