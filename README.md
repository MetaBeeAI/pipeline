# MetaBeeAI LLM Project

A project for processing and analyzing scientific papers about bee populations using Large Language Models and Vision AI.

## Project Structure 

```
MetaBeeAI_LLM/
├── process_pdfs/                    # Convert full text PDFs to structured JSON using LandingAI Vision Agentic Document Extraction
│   ├── split_pdf.py                 # PDF splitting and processing utilities
│   └── va_process_papers.py         # Vision AI pipeline for document analysis
│
├── metabeeai_llm/                   # Pipeline for using LLMs to obtain answers to specific questions from papers
│   ├── __init__.py                  # Package initialization
│   ├── llm_pipeline.py              # Main LLM processing pipeline
│   ├── json_multistage_qa.py        # Multi-stage question-answering with LLMs
│   ├── process_llm_output.py        # Process and structure LLM outputs
│   ├── unique_chunk_id.py           # Ensure unique chunk IDs across documents
│   ├── merger.py                    # Merge and combine processed data
│   ├── synthesis.py                 # Data synthesis and summarization
│   └── questions.yml                # Configuration file defining questions for LLM processing
│
├── llm_review_software/             # Combines answers with chunks and PDFs, provides GUI for reviewing LLM output
│   ├── beegui.py                    # Main GUI application for reviewing and annotating
│   ├── annotator.py                 # PDF annotation utilities
│   ├── merger.py                    # Data merging and combination
│   └── synthesis.py                 # Data synthesis and reporting
│
├── structured_datatable/             # Uses LLM answers to create condensed data tables (CSV/JSON)
│   └── field_examples.yaml          # Configuration for field extraction examples
│
├── data/                            # Data directory containing processed papers
│   └── papers/                      # Individual paper directories with processed data
│
├── metabeeai-frontend/              # Next.js frontend application
│   └── src/                         # Frontend source code
│

├── schema_config.yaml               # Schema configuration for data extraction
├── requirements.txt                  # Python dependencies
├── setup.py                         # Package installation configuration
└── README.md                        # This file
```

## Component Overview

### 1. **process_pdfs**
Converts full text PDFs to structured JSON using LandingAI Vision Agentic Document Extraction. This component:
- Splits PDFs into manageable chunks
- Uses Vision AI to extract text and structure from PDF pages
- Outputs structured JSON for further processing

### 2. **metabeeai_llm**
Core pipeline for using Large Language Models to obtain answers to specific questions from the papers. This component:
- Processes the structured JSON from Vision AI
- Uses LLMs to answer questions defined in `metabeeai_llm/questions.yml`
- Outputs structured answers for each paper

### 3. **llm_review_software**
Combines the LLM answers with document chunks and PDFs, providing a GUI for reviewing LLM output and providing comments. This component:
- Integrates all processed data (PDFs, chunks, LLM answers)
- Provides interactive GUI for human review and annotation
- Saves review outputs alongside original data for further processing

### 4. **structured_datatable**
Uses the LLM answers and further condenses them (with another LLM pipeline) to output a data table (CSV) or JSON structured output. This component:
- Takes the reviewed LLM outputs
- Applies additional LLM processing for data condensation
- Outputs structured data tables in CSV or JSON format

## Required Input Data

Before using the MetaBeeAI pipeline, you need to prepare your dataset of papers. We recommend using **ASReview** for ML-assisted paper selection and scoring.

### 1. **Paper Selection with ASReview**

**ASReview** is a machine learning tool that helps you efficiently screen papers for systematic reviews:

- **Input**: Upload abstracts and titles of candidate papers
- **Process**: ASReview uses active learning to score papers based on your initial decisions
- **Output**: Ranked list of papers with relevance scores
- **Benefit**: Reduces manual screening time by 95% while maintaining quality

**Installation and Setup:**
```bash
pip install asreview
asreview lab
```

### 2. **Data Structure Requirements**

After selecting papers with ASReview, organize your data as follows:

```
data/
├── papers/                          # Main papers directory
│   ├── 001/                        # Paper subfolder (unique number)
│   │   ├── 001_main.pdf            # Full-text PDF (rename as needed)
│   │   └── ...                     # Other paper files
│   ├── 002/                        # Second paper
│   │   ├── 002_main.pdf            # Full-text PDF
│   │   └── ...                     # Other paper files
│   └── ...                         # Additional papers
└── included_papers.csv              # ASReview output with paper metadata
```

**Folder Naming Convention:**
- Use **numbers** (001, 002, 003, ...)
- Each number should be **unique** across all papers
- Numbers should be **sequential** for easy organization

**PDF Requirements:**
- **One PDF per subfolder**
- **Full-text PDFs** (not just abstracts)
- **Rename PDFs** to match folder structure (e.g., `001_main.pdf`)
- **High-quality scans** for better Vision AI processing

### 3. **Paper Metadata CSV**

ASReview exports a CSV file containing metadata for included papers. Place this file in the `data/` folder:

**Required CSV Columns:**
- **paper_id**: Must match the subfolder names (001, 002, 003, ...)
- **title**: Paper title
- **authors**: Author names
- **journal**: Journal name
- **year**: Publication year
- **doi**: Digital Object Identifier (if available)
- **abstract**: Paper abstract (if available)

**Example CSV Structure:**
```csv
paper_id,title,authors,journal,year,doi,abstract
001,"Effects of pesticides on bee populations","Smith et al.","Nature",2023,"10.1038/...","This study examines..."
002,"Neonicotinoid impact on pollinators","Jones et al.","Science",2023,"10.1126/...","Research shows that..."
```

### 4. **Data Validation**

Before running the pipeline, ensure:

- [ ] **PDF files exist** in each numbered subfolder
- [ ] **PDFs are readable** and contain full text
- [ ] **CSV metadata** is in the `data/` folder
- [ ] **Paper IDs match** between folders and CSV
- [ ] **No duplicate numbers** across subfolders
- [ ] **File permissions** allow reading by the pipeline

**Quick Validation Command:**
```bash
# Check folder structure
ls -la data/papers/

# Verify PDF files exist
find data/papers/ -name "*.pdf" -type f

# Check CSV file
ls -la data/included_papers.csv
```

## Configuration

The MetaBeeAI pipeline uses environment variables to configure data directories and API keys. This allows you to customize the pipeline for your specific setup.

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Data Directory Configuration
METABEEAI_DATA_DIR=/path/to/your/papers/directory

# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LANDING_AI_API_KEY=your_landing_ai_api_key_here
```

### Data Directory Configuration

The `METABEEAI_DATA_DIR` environment variable controls where the pipeline looks for and saves data:

- **If not set**: Defaults to `"data/papers"` relative to the project root
- **If set**: Points to your custom papers directory

**Example configurations:**
```bash
# macOS/Linux
METABEEAI_DATA_DIR=/Users/username/Documents/research/papers

# Windows
METABEEAI_DATA_DIR=C:\Users\username\Documents\research\papers

# Linux
METABEEAI_DATA_DIR=/home/username/research/papers
```

**Directory Structure Expected:**
```
your_custom_papers_directory/
├── papers/                          # Paper subfolders
│   ├── 001/                        # Paper 001
│   │   ├── 001_main.pdf            # Full-text PDF
│   │   └── pages/                  # Processed pages
│   ├── 002/                        # Paper 002
│   │   ├── 002_main.pdf            # Full-text PDF
│   │   └── pages/                  # Processed pages
│   └── ...                         # Additional papers
├── included_papers.csv              # Paper metadata
├── logs/                           # Processing logs
└── output/                         # Generated outputs
```

### Benefits of Environment Configuration

1. **Flexibility**: Point to any directory on your system
2. **Portability**: Easy to move between different machines
3. **Multi-Project Support**: Use different directories for different projects
4. **Clean Separation**: Keep data separate from code
5. **Easy Sharing**: Share code without sharing data paths

### Setup Instructions

1. **Copy the example file:**
   ```bash
   cp env.example .env
   ```

2. **Edit `.env` with your paths:**
   ```bash
   # Edit the file with your preferred text editor
   nano .env
   # or
   code .env
   ```

3. **Set your data directory:**
   ```bash
   METABEEAI_DATA_DIR=/path/to/your/papers
   ```

4. **Add your API keys:**
   ```bash
   OPENAI_API_KEY=sk-your-actual-key-here
   ANTHROPIC_API_KEY=your-actual-key-here
   LANDING_AI_API_KEY=your-actual-key-here
   ```

**Note**: The `.env` file is not tracked in git, so your API keys and custom paths remain private.

## LLM Benchmarking and Dataset Generation

The pipeline includes tools for generating test datasets and evaluating LLM outputs using DeepEval:

### Data Merging and Dataset Generation
- **Location**: `llm_benchmarking/merge-answers.py`
- **Purpose**: Combines LLM answers with human reviewer answers for dataset creation
- **Input Sources**:
  - LLM answers: `papers/{paper_id}/answers.json`
  - Reviewer answers: Either local `answers_extended.json` files or external reviewer databases
- **Output**: Merged JSON files in `final_merged_data/` folder

### Dataset Generation Workflow
1. **Data Merging**: Combine LLM and reviewer answers using `merge_answers.py`
2. **Test Dataset Creation**: Generate evaluation datasets using `test_dataset_generation.py`
3. **Reviewer Comparison**: Create reviewer comparison datasets using `reviewer_dataset_generation.py`
4. **DeepEval Assessment**: Evaluate datasets using various DeepEval metrics

### Usage Examples
```bash
# Use local answers_extended.json files (default)
python llm_benchmarking/merge-answers.py

# Use external reviewer database
python llm_benchmarking/merge-answers.py --reviewer-db /path/to/reviewers
```

### LLM Evaluation with DeepEval

The pipeline includes comprehensive evaluation tools using DeepEval for assessing LLM performance:

#### Test Dataset Generation
- **Location**: `llm_benchmarking/test_dataset_generation.py`
- **Purpose**: Generates test datasets using reviewer answers to various questions about bee research papers
- **Features**:
  - **Input**: Questions asked about bee research papers
  - **Expected Output**: Human reviewer answers (gold standard)
  - **Context**: Relevant text chunks from original papers
  - **Metadata**: Paper ID, question type, reviewer, and quality rating
  - **Output Formats**: Both JSON and CSV for flexibility
  - **Output Location**: `llm_benchmarking/test-datasets/` folder

#### Reviewer Comparison Dataset Generation
- **Location**: `llm_benchmarking/reviewer_dataset_generation.py`
- **Purpose**: Generates test datasets comparing answers from two different reviewers for the same questions
- **Features**:
  - **Input**: Questions asked about bee research papers
  - **Expected Output**: Reviewer 1 answers (used as gold standard)
  - **Actual Outputs**: Reviewer 2 answers (for comparison)
  - **Metadata**: Paper ID, question type, and both reviewers' ratings
  - **Use Cases**: Evaluating inter-reviewer agreement, training models to identify consensus vs. disagreement
  - **Output Formats**: Both JSON and CSV for flexibility
  - **Output Location**: `llm_benchmarking/test-datasets/` folder

#### Traditional DeepEval Metrics
- **Location**: `llm_benchmarking/deepeval_benchmarking.py`
- **Purpose**: Evaluates LLM outputs using faithfulness, contextual precision, and contextual recall metrics
- **Features**: 
  - **FaithfulnessMetric**: Measures consistency between actual/expected outputs while validating against paper context
  - **ContextualPrecisionMetric**: Assesses relevance and focus of retrieved context
  - **ContextualRecallMetric**: Evaluates completeness of context coverage
  - Batch processing, incremental saving, retry logic, and cost optimization

#### G-Eval for Correctness Assessment
- **Location**: `llm_benchmarking/deepeval_GEval.py`
- **Purpose**: Uses G-Eval metrics to assess correctness, completeness, and accuracy of LLM outputs
- **Features**: 
  - **Correctness**: Strict evaluation of output accuracy against expected results
  - **Completeness**: Assessment of coverage of key points
  - **Accuracy**: Evaluation of information accuracy and alignment
  - Uses GPT-4o by default for best evaluation quality
  - Same robust batch processing and error handling as traditional benchmarking

#### Reviewer Comparison Evaluation
- **Location**: `llm_benchmarking/deepeval_reviewers.py`
- **Purpose**: Evaluates reviewer comparison dataset using both context-free and context-aware metrics
- **Features**:
  - **FaithfulnessMetric**: Measures how faithful reviewer 2 answers are to reviewer 1 answers (requires paper context with `--add-context`)
  - **G-Eval Correctness**: Strict evaluation of reviewer 2 accuracy against reviewer 1
  - **G-Eval Completeness**: Assessment of reviewer 2 coverage of reviewer 1 key points
  - **G-Eval Accuracy**: Evaluation of reviewer 2 information accuracy vs reviewer 1
  - **Flexible Context**: Can run with or without paper context depending on needs
  - **Faithfulness-Only Mode**: `--faithfulness-only` flag for efficient context-aware evaluation
  - **Reviewer Analysis**: Provides insights into inter-reviewer agreement patterns

#### Test Dataset Questions Covered
The test dataset includes 7 question types:
1. **bee_species**: "What species of bee(s) were tested?"
2. **pesticides**: "What pesticide(s) were used in this study, and what was the dose, exposure method and duration of exposure of the pesticide(s)?"
3. **additional_stressors**: "Were any additional stressors or combination used (like temperature, parasites or pathogens, other chemicals or nutrition stress)?"
4. **experimental_methodology**: "What experimental methodologies was used in this paper?"
5. **significance**: "Summarize the paper's discussion regarding the importance to the field."
6. **future_research**: "Summarize the paper's discussion regarding future research directions."
7. **limitations**: "Summarize the paper's discussion regarding any limitations or barriers to research."

#### Usage Examples
```bash
# Generate test dataset from reviewer answers
python llm_benchmarking/test_dataset_generation.py

# Generate reviewer comparison dataset
python llm_benchmarking/reviewer_dataset_generation.py

# Run traditional DeepEval evaluation
python llm_benchmarking/deepeval_benchmarking.py --question bee_species

# Run G-Eval correctness evaluation (recommended for quality assessment)
python llm_benchmarking/deepeval_GEval.py --question bee_species

# Run reviewer comparison evaluation
python llm_benchmarking/deepeval_reviewers.py --question bee_species

# Run FaithfulnessMetric only (requires context)
python llm_benchmarking/deepeval_reviewers.py --faithfulness-only --add-context --question bee_species

# Run with custom settings
python llm_benchmarking/deepeval_GEval.py --question pesticides --batch-size 25 --model gpt-4o
```

#### Evaluation Outputs
Both evaluation scripts save results in `llm_benchmarking/deepeval-results/` with:
- **JSON format**: Complete evaluation results with metadata
- **JSONL format**: Line-by-line results for easy processing
- **Timestamped filenames**: Unique identification for each evaluation run
- **Incremental saving**: Results saved after each batch to prevent data loss

#### Test Dataset Structure
The generated test dataset follows this structure:
```json
{
  "input": "What species of bee(s) were tested?",
  "expected_output": "Honey bees (Apis mellifera) were tested.",
  "actual_outputs": "Honey bees (*Apis mellifera*), specifically the Carniolan honey bee (*Apis mellifera carnica*), were tested in the study.",
  "context": ["Text chunk 1 from the paper...", "Text chunk 2 from the paper..."],
  "metadata": {
    "paper_id": "594",
    "question_id": "bee_species",
    "reviewer": "AJ",
    "rating": 10
  }
}
```

#### Data Sources
- **Questions**: From `llm_benchmarking/llm_questions.txt`
- **Reviewer Answers**: From `final_merged_data/*_merged.json` files
- **Context**: From paper pages and merged chunk data
- **Chunk IDs**: From paper `answers.json` files
- **Output Location**: Generated datasets are saved in `llm_benchmarking/test-datasets/` folder

### Streamlined Structure

The `llm_benchmarking/` module has been streamlined to focus on core benchmarking functions:

- **Core Functions**: Dataset generation, DeepEval evaluation, reviewer analysis, and results visualization
- **Moved Files**: `process_benchmarking.py` and related utilities have been moved to `structured_datatable/from_benchmarking_to_fix/` for future integration with the structured data pipeline

This ensures a clean separation between benchmarking (evaluation and analysis) and final data structuring (extraction and formatting).

See `llm_benchmarking/README.md` for detailed documentation on the streamlined benchmarking workflow and folder structures.

### DeepEval Results Analysis

The `deepeval_results_analysis.py` script provides comprehensive analysis and visualization of evaluation results:

- **Data Consolidation**: Merges results from multiple evaluation runs into organized files
- **Visualization**: Creates comparison plots between LLM vs reviewer and reviewer vs reviewer data
- **Statistical Analysis**: Provides standard error bars and detailed metric breakdowns
- **Output Organization**: Saves merged data and plots in structured directories

This tool enables researchers to:
- Compare performance across different evaluation approaches
- Identify patterns in LLM vs human performance
- Generate publication-ready visualizations
- Maintain organized historical evaluation data

**✅ Status**: Successfully tested and working! The script has processed 1,658 evaluation entries across 6 metrics and 6 question types, generating comprehensive visualizations and merged data files.

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:
```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
LANDING_AI_API_KEY=your_landing_key
```

## Usage

### 1. Process PDFs
```bash
# Split and process PDFs using Vision AI
python process_pdfs/va_process_papers.py
```

### 2. Run LLM Pipeline
```bash
# Process papers with LLM to answer questions
python -m metabeeai_llm.llm_pipeline
```

### 3. Review and Annotate
```bash
# Launch the review GUI
python llm_review_software/beegui.py
```

### 4. Generate Test Datasets and Evaluate
```bash
# Generate test datasets from reviewer answers
python llm_benchmarking/test_dataset_generation.py

# Generate reviewer comparison dataset
python llm_benchmarking/reviewer_dataset_generation.py

# Evaluate with DeepEval (traditional metrics)
python llm_benchmarking/deepeval_benchmarking.py --question bee_species

# Evaluate with G-Eval (correctness assessment)
python llm_benchmarking/deepeval_GEval.py --question bee_species

# Evaluate reviewer comparisons
python llm_benchmarking/deepeval_reviewers.py --question bee_species

# Analyze and visualize all evaluation results
python llm_benchmarking/deepeval_results_analysis.py
```

### 5. Generate Final Structured Output
```bash
# Create structured data tables (using moved utilities)
# Note: process_benchmarking.py has been moved to structured_datatable/from_benchmarking_to_fix/
# for future integration with the structured data pipeline
```

## Processing Pipeline

The MetaBeeAI pipeline follows a structured workflow:

### **Phase 1: Document Processing**
- **PDF Processing**: Automated PDF splitting and Vision AI analysis
- **LLM Integration**: Multi-stage question-answering with Large Language Models

### **Phase 2: Review and Annotation**
- **Interactive Review**: GUI-based review and annotation system
- **Data Merging**: Combine LLM and reviewer answers for analysis

### **Phase 3: Benchmarking and Evaluation**
- **Dataset Generation**: Create test datasets for evaluation
- **DeepEval Assessment**: Comprehensive evaluation using various metrics
- **Reviewer Analysis**: Statistical analysis of reviewer ratings and agreement
- **Results Analysis**: Consolidate and visualize evaluation results

### **Phase 4: Final Data Structuring** (Future)
- **Structured Output**: Final data extraction and formatting
- **Integration**: Connect with structured data pipeline

## Features

- **PDF Processing**: Automated PDF splitting and Vision AI analysis
- **LLM Integration**: Multi-stage question-answering with Large Language Models
- **Interactive Review**: GUI-based review and annotation system
- **Dataset Generation**: Automated creation of evaluation datasets
- **DeepEval Integration**: Comprehensive LLM output assessment
- **Modular Design**: Independent components that can be used separately
- **Comprehensive Logging**: Detailed processing logs and status updates

## Dependencies

Key dependencies include:
- **Vision AI**: `requests` for LandingAI API integration
- **LLM Processing**: `openai`, `litellm` for language model interactions
- **PDF Handling**: `PyPDF2`, `pymupdf` for PDF processing
- **GUI**: `PyQt5` for the review interface
- **Data Processing**: `pandas`, `numpy` for data manipulation
- **Web Framework**: `fastapi`, `uvicorn` for API services

See `requirements.txt` for the complete list of dependencies.

## Configuration

- **`metabeeai_llm/questions.yml`**: Define the questions to ask the LLM about each paper
- **`schema_config.yaml`**: Configure the data extraction schema
- **`field_examples.yaml`**: Provide examples for field extraction

## Notes

- API keys are required for OpenAI, Anthropic, and Landing AI services
- Processing large PDFs may take significant time
- The pipeline is designed to be modular - you can run components independently
- Check processing logs in the data directory for status updates
- The GUI requires PyQt5 and may have platform-specific requirements

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License. See the LICENSE file for details.
