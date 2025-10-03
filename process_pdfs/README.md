# PDF Processing Pipeline

This directory contains tools for processing research papers from PDF format into structured JSON data suitable for LLM analysis. The pipeline handles splitting, extraction, merging, and deduplication of text chunks.

## Directory Structure

Papers should be organized as follows:

```
Data directory (set in .env)
├── 001/                    # Paper folder (paperID as name)
│   ├── 001_main.pdf       # Main PDF file
│   └── pages/             # Generated during processing
│       ├── 001_main_p1-2.pdf
│       ├── 001_main_p3-4.pdf
│       ├── main_001_main_p1-2.pdf.json
│       ├── main_001_main_p3-4.pdf.json
│       ├── merged_v2.json
│       └── merged_v2_deduplicated.json
├── 002/
│   └── ...
```

## Processing Workflow

The pipeline consists of 4 main steps that should be executed in order:

### 1. `split_pdf.py` - PDF Splitting

Splits large PDFs into overlapping 2-page segments to work around Vision Agent's page limitations.

**Purpose:**
- Creates overlapping 2-page PDF segments from the main PDF
- Enables processing of large documents that would otherwise exceed Vision Agent limits
- Maintains content continuity through page overlaps

**Usage:**
```bash
# Process all papers in default data directory
python split_pdf.py

# Process papers in specific directory
python split_pdf.py --papers-dir /path/to/papers

# Process starting from specific paper folder
python split_pdf.py --start-folder 005
```

**Output:**
- Creates `paperID_main_p1-2.pdf`, `paperID_main_p3-4.pdf`, etc. in each paper's subdirectory

### 2. `va_process_papers.py` - Vision Agent Processing

Converts PDF segments into structured JSON using Vision Agentic Document Analysis.

**Purpose:**
- Extracts structured text chunks from PDF segments
- Identifies chunk types (headers, body text, tables, etc.)
- Provides grounding information (page numbers, bounding boxes)
- Handles API communication with Vision Agent service

**Usage:**
```bash
# Process all papers
python va_process_papers.py

# Process papers in specific directory
python va_process_papers.py --papers-dir /path/to/papers

# Resume processing from specific folder
python va_process_papers.py --start-folder 010
```

**Requirements:**
- Vision Agent API endpoint configured in environment
- API key set in `.env` file

**Output:**
- Creates `main_paperID_main_p1-2.pdf.json`, `main_paperID_main_p3-4.pdf.json`, etc.
- Each JSON contains structured chunks with metadata

### 3. Deduplication - `deduplicate_chunks.py` vs `batch_deduplicate.py`

Two options for removing duplicate text chunks:

#### `deduplicate_chunks.py` - Core Module
- **Purpose**: Core deduplication functions and single-file processing
- **Use for**: Processing individual merged files or custom workflows
- **Contains**: `deduplicate_chunks()`, `analyze_chunk_uniqueness()`, `process_merged_json_file()`

#### `batch_deduplicate.py` - Batch Processing Script
- **Purpose**: Automated batch processing of all papers
- **Use for**: Processing entire datasets efficiently
- **Features**: Progress tracking, error handling, dry-run mode, paper range selection

**Recommended Usage:**
```bash
# Batch process all papers (recommended)
python batch_deduplicate.py

# Dry run to analyze without changes
python batch_deduplicate.py --dry-run

# Process specific paper range
python batch_deduplicate.py --start-paper 1 --end-paper 50
```

**What it does:**
- Groups chunks by identical text content
- Preserves all chunk IDs from duplicates in a `chunk_ids` list
- Keeps the chunk with the most complete metadata
- Provides deduplication statistics and reports

**Output:**
- Creates `merged_v2_deduplicated.json` files
- Generates deduplication reports and statistics

### 4. `merger.py` - JSON Merging

Combines individual page JSON files into a single merged file with proper page numbering.

**Purpose:**
- Merges all page-level JSON files into one consolidated file
- Handles page number adjustments for overlapping segments
- Provides optional filtering of unwanted chunk types
- Maintains proper grounding information across merged pages

**Usage:**
```bash
# Basic merge (no filtering)
python merger.py --basepath /path/to/data

# Filter out headers and footers
python merger.py --basepath /path/to/data --filter-chunk-type header footer

# Filter multiple chunk types
python merger.py --basepath /path/to/data --filter-chunk-type header footer marginalia page_number
```

**Filtering Options:**
- `header` - Page headers
- `footer` - Page footers
- `marginalia` - Margin notes/annotations
- `page_number` - Page numbers
- `watermark` - Watermarks
- Custom chunk types as identified by Vision Agent

**Output:**
- Creates `merged_v2.json` in each paper's `pages/` directory
- Provides page and chunk count statistics

## Complete Pipeline Execution

To process papers from start to finish:

```bash
# 1. Split PDFs into 2-page segments
python split_pdf.py

# 2. Extract structured data with Vision Agent
python va_process_papers.py

# 3. Merge JSON files (with optional filtering)
python merger.py --basepath /path/to/data --filter-chunk-type header footer

# 4. Remove duplicate chunks
python batch_deduplicate.py
```

## Configuration

- **Data Directory**: Set `METABEEAI_DATA_DIR` in `.env` file
- **Vision Agent**: Configure API endpoint and key in `.env`
- **Logging**: Most scripts support verbose logging for debugging

## Output Files

- **Split PDFs**: `paperID_main_p1-2.pdf`, etc.
- **Raw JSON**: `main_paperID_main_p1-2.pdf.json`, etc.
- **Merged JSON**: `merged_v2.json`
- **Final Output**: `merged_v2_deduplicated.json`

See `README_deduplication.md` for detailed information about the deduplication system.