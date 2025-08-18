# LLM Benchmarking and Data Merging

This module handles the merging of LLM-generated answers with human reviewer answers for benchmarking and analysis purposes.

## Overview

The `merge_answers.py` script combines:
1. **LLM Answers**: Extracted from either the same folder as `answers_extended.json` or from a separate data directory
2. **Reviewer Answers**: From `answers_extended.json` files in a nested directory structure

## Data Sources

### LLM Answers
The script searches for LLM answers in two possible locations:
1. **Same folder as answers_extended.json**: `{reviewer_db}/{reviewer_initials}/{paper_id}/answers.json`
2. **Separate data directory**: `{data_dir}/{paper_id}/answers.json`

**Format**: JSON files containing LLM-generated responses to questions
**Structure**: Nested question paths (e.g., `bee_and_pesticides.bee_species`)

### Reviewer Answers
- **Location**: `{reviewer_db}/{reviewer_initials}/{paper_id}/answers_extended.json`
- **Format**: JSON files containing human reviewer annotations with ratings
- **Reviewer Names**: Extracted from folder names (e.g., "AB", "HS", "NN")
- **Use Case**: Multiple reviewers working independently in separate folders

## Expected Folder Structure

The script expects a nested directory structure where each reviewer has their own folder:

```
reviewer_database/
├── AB/                              # Reviewer initials (Alice Brown)
│   ├── 729/
│   │   ├── 729_main.pdf
│   │   ├── answers.json             # LLM answers (optional)
│   │   ├── answers_extended.json    # Reviewer answers with ratings
│   │   └── pages/
│   ├── 731/
│   │   ├── 731_main.pdf
│   │   ├── answers.json
│   │   ├── answers_extended.json
│   │   └── pages/
│   └── ...
├── HS/                              # Reviewer initials (Henry Smith)
│   ├── 729/
│   │   ├── 729_main.pdf
│   │   ├── answers.json
│   │   ├── answers_extended.json
│   │   └── pages/
│   ├── 731/
│   │   ├── 731_main.pdf
│   │   ├── answers.json
│   │   ├── answers_extended.json
│   │   └── pages/
│   └── ...
└── ...
```

**Note**: The same paper ID can appear in multiple reviewer folders, allowing for multiple independent reviews.

## Usage

The script requires two main parameters:

```bash
python merge_answers.py \
  --reviewer-db /path/to/reviewer/database \
  --data-dir /path/to/data/directory
```

### Command Line Options
- `--reviewer-db PATH`: **Required** - Path to reviewer database folder containing reviewer initials subfolders
- `--data-dir PATH`: **Required** - Path to data directory containing paper folders with LLM answers
- `--output-dir PATH`: **Optional** - Output directory for merged JSON files (default: `final_merged_data`)

### Example
```bash
python merge_answers.py \
  --reviewer-db "/Users/user/OneDrive/Desktop/MetaBeeAI/LLM_reviewer_output_Aug2025" \
  --data-dir "/Users/user/Documents/MetaBeeAI/pipeline/data/papers" \
  --output-dir "final_merged_data"
```

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
  "729": {
    "answer_llm": "LLM-generated answer text",
    "answer_rev1": "Reviewer answer text",
    "rev1": "AB",
    "rev1_rating": 10
  },
  "731": {
    "answer_llm": "LLM-generated answer text",
    "answer_rev1": "Reviewer answer text",
    "rev1": "AB",
    "rev1_rating": 8,
    "answer_rev2": "Alternative reviewer answer text",
    "rev2": "HS",
    "rev2_rating": 9
  }
}
```

**Note**: Papers reviewed by multiple reviewers will have `rev2`, `rev3`, etc. entries automatically added.

## Workflow for Multiple Reviewers

### Step 1: Set Up Reviewer Folders
1. Create a main reviewer database folder
2. Create subfolders with reviewer initials (e.g., "AB", "HS", "NN")
3. Each reviewer copies the full paper structure into their folder

### Step 2: Review Process
1. Each reviewer works independently in their assigned folder
2. Reviewers create `answers_extended.json` files with their answers and ratings
3. Optionally, reviewers can include `answers.json` files in the same folders
4. All reviewers work on the same set of papers

### Step 3: Merge Data
```bash
python merge_answers.py \
  --reviewer-db /path/to/reviewer/database \
  --data-dir /path/to/data/directory
```

### Step 4: Review Output
The script automatically:
- Combines multiple reviewer answers for the same paper
- Preserves all reviewer ratings
- Creates separate output files for each question type

## Configuration

The script requires explicit paths to be specified:
- **Reviewer Database**: Path to folder containing reviewer initials subfolders
- **Data Directory**: Path to folder containing paper folders with LLM answers
- **Output Directory**: Where merged JSON files will be saved (default: `final_merged_data`)

## Error Handling

The script includes robust error handling:
- Missing files are logged but don't stop processing
- Invalid JSON files are skipped with error messages
- Empty or missing answers fall back to LLM answers
- Detailed logging for debugging

## Dependencies

- Standard Python libraries: `os`, `json`, `argparse`, `pathlib`, `collections`
- No external dependencies required
