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

### 4. Generate Structured Output
```bash
# Create structured data tables
python -m metabeeai_llm.process_llm_output --start 1 --end 10
```

## Features

- **PDF Processing**: Automated PDF splitting and Vision AI analysis
- **LLM Integration**: Multi-stage question-answering with Large Language Models
- **Interactive Review**: GUI-based review and annotation system
- **Data Structuring**: Automated conversion to structured formats (CSV/JSON)
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
