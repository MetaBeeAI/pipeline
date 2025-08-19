# LLM Benchmarking and Data Merging

This module handles the merging of LLM-generated answers with human reviewer answers for benchmarking and analysis purposes.

## Scripts Overview

### `merge_answers.py`
Combines LLM-generated answers with human reviewer answers from multiple reviewers, creating merged datasets for benchmarking analysis.

### `process_benchmarking.py`
Extracts structured data from merged JSON files using LLM-based extraction, comparing LLM, rev1, and rev2 answers side-by-side with optimization for identical responses.

### `reviewer_rating.py`
Analyzes reviewer ratings and agreement across different question types and individual reviewers, generating statistical summaries and visualizations.

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

---

# Reviewer Rating Analysis

The `reviewer_rating.py` script provides comprehensive analysis of reviewer ratings and agreement across different question types and individual reviewers.

## Overview

This script analyzes the merged reviewer data to generate:
1. **Question-wise average ratings** with standard errors
2. **Reviewer agreement metrics** (differences between rev1 and rev2 ratings)
3. **Individual reviewer performance** statistics

## Features

### Data Filtering
- Automatically filters out invalid ratings (rating = 0)
- Only processes papers with valid reviewer ratings (1-10 scale)
- Handles missing or incomplete data gracefully

### Statistical Analysis
- Calculates mean ratings with standard errors
- Computes reviewer agreement as absolute differences between ratings
- Aggregates individual reviewer performance across all questions

### Visualization
- Generates publication-quality bar charts with error bars
- Color-coded plots for different analysis types
- Automatic label placement and formatting

## Usage

### Basic Usage
```bash
cd /path/to/pipeline
python llm_benchmarking/reviewer_rating.py
```

### Requirements
- Must be run from the main `pipeline` directory
- Requires `final_merged_data/` folder with merged JSON files
- Automatically creates `llm_benchmarking/analyses/` output directory

## Output Files

The script generates three PNG files in `llm_benchmarking/analyses/`:

### 1. `avg-rev-ratings.png`
**Question-wise Average Ratings**
- X-axis: Question types (bee_species, pesticides, etc.)
- Y-axis: Average rating scores (1-10 scale)
- Error bars: Standard errors
- Color: Sky blue bars with navy edges

### 2. `avg-rev-agreement.png`
**Reviewer Agreement Analysis**
- X-axis: Question types
- Y-axis: Average rating differences (lower = better agreement)
- Error bars: Standard errors
- Color: Light coral bars with dark red edges
- **Note**: Lower values indicate better agreement between reviewers

### 3. `avg-score-per-reviewer.png`
**Individual Reviewer Performance**
- X-axis: Reviewer initials (AJ, LH, EA, etc.)
- Y-axis: Average rating scores across all questions
- Error bars: Standard errors
- Color: Light green bars with dark green edges
- **Note**: Reviewers are sorted by average rating (highest to lowest)

## Data Structure Requirements

The script expects merged JSON files with the following structure:

```json
{
  "paper_id": {
    "answer_llm": "LLM-generated answer",
    "answer_rev1": "Reviewer 1 answer",
    "rev1": "AJ",
    "rev1_rating": 10,
    "answer_rev2": "Reviewer 2 answer",
    "rev2": "LH",
    "rev2_rating": 9
  }
}
```

## Statistical Methods

### Rating Calculations
- **Mean Rating**: Arithmetic mean of all valid ratings (1-10)
- **Standard Error**: `σ/√n` where σ is standard deviation and n is sample size
- **Sample Count**: Number of valid ratings included in calculations

### Agreement Calculations
- **Agreement Score**: `|rev1_rating - rev2_rating|` for papers with both reviewers
- **Lower scores** indicate better agreement between reviewers
- **Zero difference** means perfect agreement

### Data Filtering
- **Valid Ratings**: Only ratings 1-10 are included
- **Invalid Ratings**: Ratings of 0 are automatically excluded
- **Missing Data**: Papers without complete reviewer data are skipped

## Console Output

The script provides detailed console output including:

```
Starting Reviewer Rating Analysis...
Loading merged data...
Loaded data for 7 question types: ['limitations', 'future_research', 'significance', 'experimental_methodology', 'additional_stressors', 'pesticides', 'bee_species']

Calculating question statistics...
Processing limitations...
  - limitations: Avg Rating = 8.45 ± 0.12 (n=156)
  - limitations: Avg Agreement = 1.23 ± 0.15 (n=78)

Generating plots...
Question ratings plot saved to: llm_benchmarking/analyses/avg-rev-ratings.png
Reviewer agreement plot saved to: llm_benchmarking/analyses/avg-rev-agreement.png
Individual reviewer ratings plot saved to: llm_benchmarking/analyses/avg-score-per-reviewer.png

Analysis complete! All plots have been saved to the analyses directory.

============================================================
SUMMARY STATISTICS
============================================================

Question-wise Average Ratings (excluding ratings = 0):
  limitations              :   8.45 ±  0.12 (n=156)
  future_research          :   7.89 ±  0.15 (n=142)
  ...

Question-wise Reviewer Agreement (lower = better):
  limitations              :   1.23 ±  0.15 (n= 78)
  future_research          :   1.45 ±  0.18 (n= 71)
  ...

Individual Reviewer Average Ratings:
  AJ :   8.67 ±  0.08 (n=245)
  LH :   8.23 ±  0.09 (n=198)
  EA :   7.89 ±  0.11 (n=167)
```

## Dependencies

- **Required**: `numpy`, `matplotlib`, `seaborn`
- **Optional**: `pandas` (for enhanced data handling)
- **Built-in**: `json`, `os`, `pathlib`, `typing`

## Error Handling

- **Missing Data**: Gracefully handles incomplete reviewer data
- **Invalid Ratings**: Automatically filters out ratings of 0
- **File I/O**: Creates output directories if they don't exist
- **Plotting**: Handles cases where plotting libraries are unavailable

## Use Cases

### Quality Assessment
- Evaluate consistency of reviewer ratings across question types
- Identify questions that may need clarification or revision
- Assess overall quality of the review process

### Reviewer Performance
- Compare individual reviewer consistency and rigor
- Identify potential training needs for specific reviewers
- Ensure balanced review workload distribution

### Research Validation
- Validate the robustness of LLM-generated answers
- Assess inter-rater reliability in the review process
- Support quality metrics for research publications

---

# Structured Data Extraction

The `process_benchmarking.py` script extracts structured data from merged JSON files using LLM-based extraction, with intelligent optimization for identical responses.

## Overview

This script processes one question type at a time, extracting structured information from LLM, rev1, and rev2 answers. It includes optimization to reduce LLM API calls when answers are identical or when reviewers are missing.

## Features

### LLM-Based Extraction
- Uses GPT-4o-mini for semantic data extraction
- Structured prompts designed to prevent hallucination
- Extracts specific fields based on question type (bee_species, pesticides, additional_stressors)

### Smart Optimization
- Detects identical answers between LLM, rev1, and rev2
- Processes identical responses only once to save API calls
- Handles missing reviewers gracefully (no false data copying)

### Question-Specific Processing
- **bee_species**: Extracts scientific names, common names, or both as provided
- **pesticides**: Structured extraction of compound names, doses, exposure methods, timing
- **additional_stressors**: Pathogens, parasites, environmental stressors (excluding pesticides)

## Usage

### Basic Usage
```bash
cd /path/to/pipeline
python llm_benchmarking/process_benchmarking.py --question bee_species
```

### Command Line Options
- `--question QUESTION`: Specific question to process (e.g., bee_species, pesticides)
- `--max-papers N`: Maximum number of papers to process (useful for testing)
- `--interactive`: Run in interactive mode with menu selection
- `--data-dir PATH`: Data directory path (default: `llm_benchmarking/final_merged_data`)
- `--output-dir PATH`: Output directory (default: `extracted_data`)

### Examples
```bash
# Process bee_species question for all papers
python llm_benchmarking/process_benchmarking.py --question bee_species

# Test with first 10 papers
python llm_benchmarking/process_benchmarking.py --question bee_species --max-papers 10

# Interactive mode
python llm_benchmarking/process_benchmarking.py --interactive
```

## Output

### Files Generated
- **CSV**: `{question}_extracted.csv` - Tabular format for analysis
- **JSON**: `{question}_extracted.json` - Structured format for further processing

### Output Structure
```json
{
  "paper_id": "594",
  "question_type": "bee_species",
  "llm_answer": "Honey bees (Apis mellifera), specifically the Carniolan honey bee (Apis mellifera carnica)...",
  "rev1_answer": "Honey bees (Apis mellifera), specifically the Carniolan honey bee (Apis mellifera carnica)...",
  "rev1_rating": 10,
  "rev1_reviewer": "AJ",
  "rev2_answer": "The bee species tested was Apis mellifera. The subspecies was Apis mellifera carnica.",
  "rev2_rating": 10,
  "rev2_reviewer": "LH",
  "extracted_llm": "Apis mellifera, Apis mellifera carnica",
  "extracted_rev1": "Apis mellifera, Apis mellifera carnica",
  "extracted_rev2": "Apis mellifera, Apis mellifera carnica"
}
```

## Optimization Features

### Identical Answer Detection
- **All identical**: Process once, copy to all three fields
- **Partial optimization**: Process identical pair once, different answer separately
- **No optimization**: Process all three answers separately

### Missing Reviewer Handling
- Only processes answers when reviewers actually exist
- No false data copying for missing rev1/rev2 reviewers
- Empty strings for missing reviewer data

### Statistics Tracking
```
Optimization stats:
  - Papers with all identical answers (processed once): 5
  - Papers with partial optimization (2 identical, 1 different): 0
  - Papers with no optimization (all different): 0
  - LLM API calls saved: 10
```

## Configuration

### Schema Files
- `benchmarking_schema.yaml`: Defines extraction structure and field types
- `benchmarking_field_examples.yaml`: Provides examples for LLM prompts

### LLM Settings
- **Model**: GPT-4o-mini
- **Temperature**: 0 (deterministic output)
- **Max tokens**: 500
- **System prompt**: JSON-only output, no explanations

## Error Handling

- **LLM failures**: Graceful fallback to simple text extraction
- **Missing data**: Handles null/empty answers appropriately
- **JSON parsing**: Robust handling of malformed LLM responses
- **File I/O**: Creates output directories automatically

## Dependencies

- **Required**: `openai`, `pyyaml`, `pandas`
- **Built-in**: `json`, `os`, `pathlib`, `argparse`, `re`

## Use Cases

### Data Analysis
- Compare extraction quality between LLM and human reviewers
- Identify systematic differences in answer interpretation
- Generate structured datasets for further research

### Quality Control
- Validate LLM extraction against human annotations
- Assess consistency of data extraction across reviewers
- Support automated data processing pipelines

### Research Efficiency
- Reduce manual data extraction time
- Standardize data formats across studies
- Enable large-scale meta-analyses
