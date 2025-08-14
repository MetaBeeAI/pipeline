# LLM Benchmarking and Data Merging

This module handles the merging of LLM-generated answers with human reviewer answers for benchmarking and analysis purposes.

## Overview

The `merge-answers.py` script combines:
1. **LLM Answers**: Extracted from `papers/{paper_id}/answers.json` files
2. **Reviewer Answers**: From either local `answers_extended.json` files or external reviewer databases

## Data Sources

### LLM Answers
- **Location**: `papers/{paper_id}/answers.json` (within the configured data directory)
- **Format**: JSON files containing LLM-generated responses to questions
- **Structure**: Nested question paths (e.g., `bee_and_pesticides.bee_species`)

### Reviewer Answers

#### Option 1: Local answers_extended.json Files
- **Location**: `papers/{paper_id}/answers_extended.json` (within the configured data directory)
- **Format**: JSON files containing human reviewer annotations
- **Reviewer Names**: Set to "NA" (not applicable for local files)
- **Use Case**: Single reviewer working directly in the main data folder

#### Option 2: External Reviewer Database
- **Location**: User-specified path to reviewer database
- **Format**: Hierarchical folder structure with reviewer initials
- **Reviewer Names**: Extracted from folder names (e.g., "JD", "SM")
- **Use Case**: Multiple reviewers working independently

## Expected Folder Structures

### For Local answers_extended.json (Default)
```
data/
├── papers/
│   ├── 001/
│   │   ├── 001_main.pdf
│   │   ├── answers.json              # LLM answers
│   │   ├── answers_extended.json     # Reviewer answers
│   │   └── pages/
│   ├── 002/
│   │   ├── 002_main.pdf
│   │   ├── answers.json
│   │   ├── answers_extended.json
│   │   └── pages/
│   └── ...
└── included_papers.csv
```

### For External Reviewer Database
```
reviewer_database/
├── JD/                              # Reviewer initials (John Doe)
│   ├── 001/
│   │   ├── 001_main.pdf
│   │   ├── answers.json             # Reviewer answers
│   │   └── pages/
│   ├── 002/
│   │   ├── 002_main.pdf
│   │   ├── answers.json
│   │   └── pages/
│   └── ...
├── SM/                              # Reviewer initials (Sarah Miller)
│   ├── 001/
│   │   ├── 001_main.pdf
│   │   ├── answers.json
│   │   └── pages/
│   ├── 002/
│   │   ├── 002_main.pdf
│   │   ├── answers.json
│   │   └── pages/
│   └── ...
└── ...
```

## Usage

### Basic Usage (Local answers_extended.json)
```bash
python merge-answers.py
```

### Using External Reviewer Database
```bash
python merge-answers.py --reviewer-db /path/to/reviewer/database
```

### Command Line Options
- `--use-extended`: Use local answers_extended.json files (default: True)
- `--reviewer-db PATH`: Path to external reviewer database (overrides --use-extended)

## Output

The script generates merged JSON files in the `final_merged_data/` folder:

```
final_merged_data/
├── bee_species_merged.json
├── experimental_methodology_merged.json
├── pesticide_types_merged.json
└── ...
```

### Output Format
Each merged file contains data for all papers with the following structure:

```json
{
  "001": {
    "answer_llm": "LLM-generated answer text",
    "answer_rev1": "Reviewer answer text",
    "rev1": "Reviewer initials or 'NA'"
  },
  "002": {
    "answer_llm": "LLM-generated answer text",
    "answer_rev1": "Reviewer answer text",
    "rev1": "Reviewer initials or 'NA'"
  }
}
```

## Workflow for Multiple Reviewers

### Step 1: Set Up Reviewer Folders
1. Create a main "reviewer" folder
2. Create subfolders with reviewer initials (e.g., "JD", "SM", "AB")
3. Each reviewer copies the full paper structure into their folder

### Step 2: Review Process
1. Each reviewer works independently in their assigned folder
2. Reviewers use the same folder structure: `{paper_id}/answers.json`
3. All reviewers work on the same set of papers

### Step 3: Merge Data
```bash
python merge-answers.py --reviewer-db /path/to/reviewer/folder
```

## Configuration

The script automatically uses the centralized configuration system:
- **Data Directory**: Set via `METABEEAI_DATA_DIR` environment variable
- **Default**: `data/papers` if no environment variable is set
- **Output**: Saved to `{data_dir}/final_merged_data/`

## Error Handling

The script includes robust error handling:
- Missing files are logged but don't stop processing
- Invalid JSON files are skipped with error messages
- Empty or missing answers fall back to LLM answers
- Detailed logging for debugging

## Dependencies

- `config.py` (centralized configuration)
- Standard Python libraries: `os`, `json`, `argparse`, `pathlib`
- Environment variable support via `python-dotenv`
