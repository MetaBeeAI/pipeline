# MetaBeeAI Pipeline

A comprehensive pipeline for extracting, analyzing, and benchmarking structured information from scientific literature using Large Language Models and Vision AI.

## Overview

The MetaBeeAI pipeline transforms PDF scientific papers into structured, analyzable data through five main stages:

1. **PDF Processing** - Convert PDFs to structured JSON chunks
2. **LLM Question Answering** - Extract information using LLMs
3. **Human Review** - Validate and annotate LLM outputs
4. **Benchmarking** - Evaluate LLM performance against human reviewers
5. **Data Analysis** - Analyze trends and relationships in extracted data

---

## Prerequisites

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# Or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```bash
# Copy example file
cp env.example .env

# Edit .env and add your settings:
# METABEEAI_DATA_DIR=/path/to/your/papers
# OPENAI_API_KEY=your_openai_key
# LANDING_AI_API_KEY=your_landing_ai_key
```

The `.env` file is hidden from git for security.

### 3. Data Organization

Organize your papers in the following structure:

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
- Each paper in a numbered folder (001, 002, 003, etc.)
- PDF named `{folder_number}_main.pdf`
- Full-text PDFs (not just abstracts)

---

## Complete Pipeline Workflow

### STAGE 1: PDF Processing

**Purpose**: Convert PDF papers into structured JSON chunks with text and metadata.

**Location**: `process_pdfs/`

**Steps**:

```bash
# Option A: Run complete pipeline (recommended)
cd process_pdfs
python process_all.py --start 1 --end 10

# Option B: Merge-only mode (if PDFs already processed)
python process_all.py --merge-only --start 1 --end 10

# Option C: Process all papers in directory
python process_all.py
```

**What happens**:
1. **Split PDFs** - Creates overlapping 2-page segments
2. **Vision API** - Extracts text and structure using Landing AI
3. **Merge JSONs** - Combines individual pages into single files
4. **Deduplicate** - Removes duplicate chunks from overlapping pages

**Output**: `papers/XXX/pages/merged_v2.json` for each paper

**Documentation**: See `process_pdfs/README.md`

---

### STAGE 2: LLM Question Answering

**Purpose**: Use LLMs to extract specific information by answering predefined questions.

**Location**: `metabeeai_llm/`

**Configuration**:

Edit `metabeeai_llm/questions.yml` to customize questions:

```yaml
bee_species:
  question: "What bee species were experimentally tested?"
  instructions:
    - "Extract species names from methodology sections"
    - "Provide full scientific names"
  output_format: "Numbered list"
  example_output:
    - "1. Apis mellifera carnica; 2. Bombus terrestris"
  max_chunks: 5
  min_score: 0.6
```

**Run the pipeline**:

```bash
cd metabeeai_llm

# Process papers 1-10
python llm_pipeline.py --start 1 --end 10

# Process all papers
python llm_pipeline.py

# Process specific directory
python llm_pipeline.py --dir /path/to/papers --start 1 --end 10
```

**What happens**:
1. Loads questions from `questions.yml`
2. For each paper, selects relevant text chunks
3. Asks LLM to answer each question based on chunks
4. Synthesizes final answers with reasoning and source citations

**Output**: `papers/XXX/answers.json` for each paper

**Documentation**: See `metabeeai_llm/README.md`

---

### STAGE 3: Human Review and Annotation

**Purpose**: Review LLM-generated answers, provide ratings, and annotate PDFs.

**Location**: `llm_review_software/`

**Tools**:

#### 3A. Review GUI
```bash
cd llm_review_software
python beegui.py
```

**Features**:
- View PDFs with highlighted source chunks
- Review LLM answers question-by-question
- Provide ratings and corrections
- Navigate between papers and questions
- Save annotations and reviews

**Output**: `answers_extended.json` files with reviewer ratings

#### 3B. PDF Annotation Tool
```bash
python annotator.py --basepath /path/to/data
```

**Features**:
- Annotates PDFs with bounding boxes showing source chunks
- Color-codes chunks by question type
- Creates annotated PDFs for verification

**Output**: Annotated PDF files

**Documentation**: Review GUI has built-in help and tooltips

---

### STAGE 4: LLM Benchmarking and Evaluation

**Purpose**: Evaluate LLM performance against human reviewer gold standards.

**Location**: `llm_benchmarking/`

**Complete benchmarking workflow**:

```bash
cd llm_benchmarking

# Run LLM v1 benchmarking (all questions)
python run_benchmarking.py --type llmv1

# Run LLM v2 benchmarking
python run_benchmarking.py --type llmv2

# Run reviewer comparison benchmarking
python run_benchmarking.py --type rev

# Run all benchmarking types
python run_benchmarking.py --type all

# Run analysis only (after benchmarking)
python run_benchmarking.py --analyze-only
```

**What happens**:

**For LLM v1 benchmarking**:
1. **Data merging** - Combines LLM and reviewer answers
2. **Dataset generation** - Creates evaluation test cases
3. **DeepEval metrics** - Evaluates faithfulness, precision, recall
4. **GEval metrics** - Evaluates correctness, completeness, accuracy
5. **Analysis** - Creates summary statistics and visualizations

**For LLM v2 benchmarking**:
1. **Merge LLM v2 data** - Adds improved LLM answers
2. **Evaluation** - Compares LLM v2 vs Reviewer 3
3. **Analysis** - Generates comparative statistics

**For Reviewer comparison**:
1. **Dataset generation** - Pairs Reviewer 1 vs Reviewer 2 answers
2. **Evaluation** - Measures inter-reviewer agreement
3. **Analysis** - Identifies consensus and disagreement patterns

**Metrics explained**:
- **Faithfulness**: How well answers align with source text
- **Contextual Precision**: Relevance of retrieved chunks
- **Contextual Recall**: Completeness of chunk coverage
- **Correctness**: Factual accuracy (binary)
- **Completeness**: Coverage of key points (continuous)
- **Accuracy**: Semantic alignment quality (continuous)

**Output**:
- `deepeval-results/*.json/jsonl` - Detailed evaluation results
- `deepeval-analyses/` - Merged data and visualizations
- `comparison_plots/` - Comparative analysis plots
- `summary_all_results_improved.csv` - Comprehensive statistics

**Documentation**: See `llm_benchmarking/README.md`

---

### STAGE 5: Data Analysis and Visualization

**Purpose**: Extract structured data and analyze trends, patterns, and relationships.

**Location**: `query_database/`

**Workflow**:

```bash
cd query_database

# Step 1: Extract structured data from LLM answers
python investigate_bee_species.py
python investigate_pesticides.py
python investigate_additional_stressors.py
python investigate_significance.py

# Step 2: Analyze trends and create visualizations
python trend_analysis.py
python network_analysis.py
```

**What each script does**:

**Data Extraction**:
- `investigate_bee_species.py` - Extracts species names, taxonomy
- `investigate_pesticides.py` - Extracts chemical names, doses, exposure methods
- `investigate_additional_stressors.py` - Extracts non-pesticide stressors
- `investigate_significance.py` - Extracts key findings and results

**Analysis**:
- `trend_analysis.py` - Analyzes co-occurrence patterns, creates bar charts
- `network_analysis.py` - Creates network visualizations showing relationships

**Output**:
- `output/*.json` - Structured data files
- `output/trend_analysis_plots/` - Trend visualizations
- `output/network_plots/` - Network diagrams
- `output/*.txt` - Statistical reports

**Documentation**: See `query_database/README.md`

---

## Quick Start Guide

### Minimal Working Example (3 papers)

```bash
# 1. Setup environment
cp env.example .env
# Edit .env with your API keys
source venv/bin/activate

# 2. Process PDFs
cd process_pdfs
python process_all.py --start 1 --end 3
cd ..

# 3. Run LLM pipeline
cd metabeeai_llm
python llm_pipeline.py --start 1 --end 3
cd ..

# 4. Extract and analyze data
cd query_database
python investigate_bee_species.py
python investigate_pesticides.py
python trend_analysis.py
cd ..

# 5. Check results
ls -lh query_database/output/
```

---

## Project Structure

```
MetaBeeAI/pipeline/
├── process_pdfs/              # STAGE 1: PDF to JSON conversion
│   ├── process_all.py         # Main pipeline runner
│   ├── split_pdf.py           # PDF splitting
│   ├── va_process_papers.py   # Vision API processing
│   ├── merger.py              # JSON merging
│   ├── batch_deduplicate.py   # Deduplication
│   └── README.md              # Documentation
│
├── metabeeai_llm/             # STAGE 2: LLM question answering
│   ├── llm_pipeline.py        # Main pipeline runner
│   ├── json_multistage_qa.py  # Q&A engine
│   ├── pipeline_config.py     # Configuration
│   ├── questions.yml          # Question definitions ⭐
│   └── README.md              # Documentation
│
├── llm_review_software/       # STAGE 3: Human review and annotation
│   ├── beegui.py              # Review GUI application
│   ├── annotator.py           # PDF annotation tool
│   └── __init__.py
│
├── llm_benchmarking/          # STAGE 4: LLM evaluation
│   ├── run_benchmarking.py    # Main benchmarking runner
│   ├── merge_answers.py       # Data merging
│   ├── test_dataset_generation.py          # Dataset creation
│   ├── deepeval_benchmarking.py            # Standard metrics
│   ├── deepeval_GEval.py                   # GEval metrics
│   ├── deepeval_llmv2.py                   # LLM v2 evaluation
│   ├── deepeval_reviewers.py               # Reviewer comparison
│   ├── analyze_deepeval_results_improved.py # Analysis
│   ├── create_comparison_plots.py          # Visualization
│   └── README.md                           # Documentation
│
├── query_database/            # STAGE 5: Data analysis
│   ├── investigate_bee_species.py          # Extract species data
│   ├── investigate_pesticides.py           # Extract pesticide data
│   ├── investigate_additional_stressors.py # Extract stressor data
│   ├── investigate_significance.py         # Extract findings
│   ├── trend_analysis.py                   # Trend analysis
│   ├── network_analysis.py                 # Network visualization
│   └── README.md                           # Documentation
│
├── data/
│   └── papers/                # Paper data (PDFs and outputs)
│
├── config.py                  # Centralized configuration
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── env.example                # Environment configuration template
└── README.md                  # This file
```

---

## Detailed Pipeline Flow

```
Raw PDFs
    ↓
[STAGE 1: process_pdfs/]
    ├── Split into 2-page segments
    ├── Extract with Vision API
    ├── Merge JSON files
    └── Deduplicate chunks
    ↓
papers/XXX/pages/merged_v2.json
    ↓
[STAGE 2: metabeeai_llm/]
    ├── Load questions from questions.yml
    ├── Select relevant chunks
    ├── Query LLM for answers
    └── Synthesize final answers
    ↓
papers/XXX/answers.json
    ↓
[STAGE 3: llm_review_software/]
    ├── Review in GUI
    ├── Rate answer quality
    ├── Provide corrections
    └── Annotate PDFs
    ↓
papers/XXX/answers_extended.json
    ↓
[STAGE 4: llm_benchmarking/]
    ├── Merge LLM + reviewer answers
    ├── Generate test datasets
    ├── Run DeepEval evaluation
    ├── Analyze results
    └── Create visualizations
    ↓
Evaluation metrics, plots, statistics
    ↓
[STAGE 5: query_database/]
    ├── Extract structured data
    ├── Analyze trends
    ├── Create network graphs
    └── Generate reports
    ↓
Final structured data + visualizations
```

---

## Step-by-Step Usage Guide

### Step 1: Process PDFs

Convert raw PDFs into structured JSON format.

```bash
cd process_pdfs

# Process all papers (complete pipeline)
python process_all.py --start 1 --end 50

# Or: Merge-only mode (if already processed with Vision API)
python process_all.py --merge-only

# Or: Process all papers in directory
python process_all.py
```

**Input**: `papers/XXX/XXX_main.pdf`

**Output**: `papers/XXX/pages/merged_v2.json`

**Time**: ~10-30 seconds per paper (Vision API step is slowest)

**Cost**: Vision API credits per page (check Landing AI pricing)

**Skip if**: You've already processed PDFs through Vision API

**See**: `process_pdfs/README.md` for details

---

### Step 2: Run LLM Pipeline

Extract information by asking LLMs specific questions about each paper.

```bash
cd metabeeai_llm

# Customize questions (optional)
nano questions.yml  # Edit question prompts, instructions, examples

# Run pipeline
python llm_pipeline.py --start 1 --end 50

# Or: Process all papers
python llm_pipeline.py
```

**Input**: `papers/XXX/pages/merged_v2.json`

**Output**: `papers/XXX/answers.json`

**Questions answered** (configurable in `questions.yml`):
- What bee species were tested?
- What pesticides were used (dose, method, duration)?
- Were additional stressors tested?
- What experimental methods were used?
- What were the key findings?
- What future research directions were suggested?

**Time**: ~30-120 seconds per paper (depends on paper length and model)

**Cost**: OpenAI API costs (GPT-4o-mini recommended for cost efficiency)

**Model configuration**: Edit `pipeline_config.py` to choose models

**See**: `metabeeai_llm/README.md` for details

---

### Step 3: Human Review and Annotation (Optional)

Review and validate LLM-generated answers with human expertise.

```bash
cd llm_review_software

# Launch review GUI
python beegui.py
```

**What the GUI provides**:
- Side-by-side PDF and answer review
- Visual highlighting of source chunks
- Question-by-question navigation
- Rating system (1-10 scale)
- Annotation and correction tools
- Save progress automatically

**Input**: 
- `papers/XXX/XXX_main.pdf` (original PDF)
- `papers/XXX/pages/merged_v2.json` (chunks)
- `papers/XXX/answers.json` (LLM answers)

**Output**: `papers/XXX/answers_extended.json` (with ratings and corrections)

**Use cases**:
- Quality assurance for critical projects
- Training data generation for improved models
- Inter-reviewer agreement studies
- Validating LLM performance

**Optional PDF annotation**:
```bash
python annotator.py --basepath /path/to/data
```
Creates annotated PDFs with bounding boxes around source chunks.

---

### Step 4: Benchmarking and Evaluation (Optional)

Evaluate LLM performance against human reviewer gold standards.

```bash
cd llm_benchmarking

# Run complete benchmarking for LLM v1
python run_benchmarking.py --type llmv1

# Run specific question type
python run_benchmarking.py --type llmv1 --question bee_species

# Compare LLM v2 vs Reviewer 3
python run_benchmarking.py --type llmv2

# Evaluate inter-reviewer agreement
python run_benchmarking.py --type rev

# Run all benchmarking types
python run_benchmarking.py --type all

# Skip dataset generation (if already done)
python run_benchmarking.py --type llmv1 --skip-dataset

# Run analysis only
python run_benchmarking.py --analyze-only
```

**Prerequisites**:
- Completed Stage 3 (human review)
- Have both LLM answers and reviewer answers

**What happens**:
1. **Data merging** - Combines LLM + reviewer answers
2. **Dataset generation** - Creates evaluation test cases
3. **DeepEval evaluation** - Runs multiple metrics
4. **Analysis** - Generates statistics and plots
5. **Visualization** - Creates comparison charts

**Metrics evaluated**:
- Faithfulness, Contextual Precision/Recall
- Correctness, Completeness, Accuracy (GEval)
- Inter-reviewer agreement scores

**Output**:
- `deepeval-results/` - Detailed evaluation results
- `deepeval-analyses/` - Visualizations and merged data
- `comparison_plots/` - Comparative analysis
- `summary_all_results_improved.csv` - Comprehensive statistics

**Time**: ~5-30 seconds per test case (depends on batch size)

**Cost**: OpenAI API costs for evaluation (gpt-4o-mini recommended)

**See**: `llm_benchmarking/README.md` for details

---

### Step 5: Data Analysis and Visualization

Analyze extracted data for trends, patterns, and relationships.

```bash
cd query_database

# Extract structured data from all papers
python investigate_bee_species.py
python investigate_pesticides.py
python investigate_additional_stressors.py
python investigate_significance.py

# Analyze trends and create visualizations
python trend_analysis.py
python network_analysis.py
```

**What happens**:

**Data extraction**:
- Parses LLM answers into structured fields
- Standardizes species names and pesticide names
- Categorizes stressors by type
- Extracts quantitative findings

**Analysis**:
- Identifies most studied species and pesticides
- Calculates co-occurrence frequencies
- Creates network graphs showing relationships
- Generates statistical reports

**Output**:
- `output/bee_species_data.json` - Species database
- `output/pesticides_data.json` - Pesticide database
- `output/additional_stressors_data.json` - Stressor database
- `output/significance_data.json` - Findings database
- `output/trend_analysis_plots/` - Bar charts and trends
- `output/network_plots/` - Network visualizations
- `output/*.txt` - Statistical reports

**Use cases**:
- Meta-analysis of research trends
- Identifying knowledge gaps
- Understanding bee-pesticide interactions
- Supporting systematic reviews

**See**: `query_database/README.md` for details

---

## Common Workflows

### Workflow A: Complete Pipeline (All Stages)

```bash
# 1. Process PDFs
cd process_pdfs && python process_all.py --start 1 --end 10 && cd ..

# 2. Run LLM pipeline
cd metabeeai_llm && python llm_pipeline.py --start 1 --end 10 && cd ..

# 3. Extract and analyze data
cd query_database
python investigate_bee_species.py
python investigate_pesticides.py
python trend_analysis.py
python network_analysis.py
cd ..
```

### Workflow B: Re-processing Existing PDFs

```bash
# Only merge and deduplicate (skip expensive Vision API)
cd process_pdfs && python process_all.py --merge-only && cd ..

# Re-run LLM with updated questions
cd metabeeai_llm && python llm_pipeline.py && cd ..
```

### Workflow C: Benchmarking Only

```bash
# Assuming you have reviewed answers
cd llm_benchmarking

# Run complete benchmarking
python run_benchmarking.py --type llmv1

# Analyze results
python run_benchmarking.py --analyze-only
```

### Workflow D: Analysis Only

```bash
# If you have answers.json files, extract and analyze
cd query_database
python investigate_bee_species.py
python investigate_pesticides.py
python trend_analysis.py
python network_analysis.py
```

---

## Configuration Files

### `questions.yml` - Question Definitions
**Location**: `metabeeai_llm/questions.yml`

**Purpose**: Define all questions the LLM will answer

**Key fields**:
- `question` - The question text
- `instructions` - Guidelines for answering
- `output_format` - Expected format
- `example_output` - Good examples
- `bad_example_output` - Examples to avoid
- `max_chunks` - How many text chunks to analyze
- `min_score` - Relevance threshold

**This is your main customization point!**

### `pipeline_config.py` - Model Configuration
**Location**: `metabeeai_llm/pipeline_config.py`

**Purpose**: Configure LLM models and processing parameters

**Options**:
- `FAST_CONFIG` - gpt-4o-mini (fast, cheap)
- `BALANCED_CONFIG` - Mixed models (recommended)
- `QUALITY_CONFIG` - gpt-4o (high quality, slower)

### `.env` - Environment Configuration
**Location**: Root directory (hidden from git)

**Required variables**:
```bash
METABEEAI_DATA_DIR=/path/to/papers
OPENAI_API_KEY=sk-...
LANDING_AI_API_KEY=...
```

---

## Cost Estimation

### For 100 papers (average 10 pages each):

| Stage | Service | Estimated Cost | Notes |
|-------|---------|----------------|-------|
| Stage 1 | Landing AI Vision API | $50-150 | Per page processing |
| Stage 2 | OpenAI (gpt-4o-mini) | $5-15 | 6 questions per paper |
| Stage 2 | OpenAI (gpt-4o) | $50-150 | Higher quality, more expensive |
| Stage 4 | OpenAI (evaluation) | $10-30 | If benchmarking |

**Cost reduction tips**:
1. Use `--merge-only` to skip re-processing PDFs
2. Use gpt-4o-mini for most tasks
3. Test on small batches first (--start 1 --end 5)
4. Skip benchmarking for production runs

---

## Troubleshooting

### "METABEEAI_DATA_DIR not set"
```bash
# Fix: Add to .env file
echo "METABEEAI_DATA_DIR=/path/to/papers" >> .env
```

### "API key not found"
```bash
# Fix: Add keys to .env file
cp env.example .env
# Edit .env and add your API keys
```

### "No merged_v2.json files found"
```bash
# Fix: Run PDF processing first
cd process_pdfs
python process_all.py --start 1 --end 10
```

### "No answers.json files found"
```bash
# Fix: Run LLM pipeline first
cd metabeeai_llm
python llm_pipeline.py --start 1 --end 10
```

### Pipeline is slow
- **PDF Processing**: Normal (Vision API is slow). Use `--merge-only` if re-running
- **LLM Pipeline**: Use FAST_CONFIG in `pipeline_config.py`
- **Benchmarking**: Reduce batch size or use gpt-4o-mini

### Out of API credits
- **Landing AI**: Check quota at Landing AI dashboard
- **OpenAI**: Check usage at platform.openai.com/usage

---

## Dependencies

Key dependencies (see `requirements.txt` for complete list):

**Core**:
- `python >= 3.8`
- `litellm` - LLM API interface
- `openai` - OpenAI API client
- `PyPDF2` - PDF manipulation
- `pymupdf` (fitz) - PDF rendering

**Data Processing**:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `pyyaml` - YAML parsing

**Visualization**:
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `networkx` - Network graphs

**GUI** (optional):
- `PyQt5` - GUI framework

**Evaluation** (optional):
- `deepeval` - LLM evaluation framework

**API Integration**:
- `requests` - HTTP requests
- `python-dotenv` - Environment variables

---

## Best Practices

### 1. Start Small
Test the pipeline on 3-5 papers before processing hundreds:
```bash
python process_all.py --start 1 --end 5
```

### 2. Customize Questions First
Review and edit `metabeeai_llm/questions.yml` before large-scale processing.

### 3. Monitor Costs
- Use gpt-4o-mini for initial runs
- Check OpenAI usage dashboard regularly
- Test prompts on small batches

### 4. Validate Data Quality
- Review a sample of LLM answers manually
- Check edge cases in benchmarking results
- Verify extracted data makes sense

### 5. Use Version Control
- Commit `questions.yml` changes
- Tag versions of pipeline configurations
- Keep changelog of prompt improvements

### 6. Backup Data
- Keep original PDFs in separate location
- Backup `answers.json` files before re-running
- Export analysis results regularly

---

## Output Files Summary

| File | Created By | Purpose |
|------|-----------|---------|
| `papers/XXX/pages/merged_v2.json` | Stage 1 | Structured PDF chunks |
| `papers/XXX/answers.json` | Stage 2 | LLM-generated answers |
| `papers/XXX/answers_extended.json` | Stage 3 | Reviewed answers with ratings |
| `deepeval-results/*.json` | Stage 4 | Evaluation results |
| `query_database/output/*.json` | Stage 5 | Extracted structured data |
| `query_database/output/*_plots/*.png` | Stage 5 | Visualizations |
| `*.txt` reports | Stages 4-5 | Statistical summaries |

---

## Support and Resources

### Documentation by Stage:
1. PDF Processing: `process_pdfs/README.md`
2. LLM Pipeline: `metabeeai_llm/README.md`
3. Review Software: Built-in GUI help
4. Benchmarking: `llm_benchmarking/README.md`
5. Data Analysis: `query_database/README.md`

### Getting Help:
1. Check relevant README for your stage
2. Review example outputs in `data/papers/`
3. Check troubleshooting sections
4. Verify environment configuration in `.env`

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{metabeeai_pipeline,
  title = {MetaBeeAI Pipeline: Automated Literature Extraction for Bee Research},
  author = {MetaBeeAI Team},
  year = {2025},
  url = {https://github.com/your-repo/metabeeai-pipeline}
}
```

---

**Last Updated**: October 2025
