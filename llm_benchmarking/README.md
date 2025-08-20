# LLM Benchmarking and Dataset Generation

This module handles the generation of test datasets from LLM-generated answers and human reviewer answers, and provides comprehensive evaluation using DeepEval for benchmarking purposes.

## Scripts Overview

### `deepeval_benchmarking.py` 
Comprehensive LLM evaluation using DeepEval with batch processing, incremental saving, and cost-effective GPT-4o-mini integration.

### `merge_answers.py`
Combines LLM-generated answers with human reviewer answers from multiple reviewers, creating merged datasets for dataset generation and benchmarking analysis.

### `reviewer_rating.py`
Analyzes reviewer ratings and agreement across different question types and individual reviewers, generating statistical summaries and visualizations.

---

## Streamlined Structure

This module has been streamlined to focus on the core benchmarking workflow:

### **Core Functions (Current):**
- **Dataset Generation**: Create test datasets for evaluation
- **DeepEval Evaluation**: Comprehensive LLM output assessment
- **Reviewer Analysis**: Statistical analysis of reviewer ratings

### **Moved to `structured_datatable/from_benchmarking_to_fix/`:**
- **`process_benchmarking.py`**: Final data extraction and structuring (moved for future integration with structured data pipeline)
- **Other utilities**: Additional processing scripts that may be useful for the final data extraction step

This streamlined structure ensures that `llm_benchmarking/` focuses on its core purpose: generating test datasets and evaluating them with DeepEval, while keeping related utilities available for future use in the structured data pipeline.

### `reviewer_dataset_generation.py`
Generates test datasets comparing answers from two different reviewers for the same questions. This script creates `rev_test_dataset.json` and `rev_test_dataset.csv` files that are essential for evaluating inter-reviewer agreement and training models to identify consensus vs. disagreement patterns.

### `deepeval_reviewers.py`
Evaluates the reviewer comparison dataset using context-free metrics from both traditional DeepEval and G-Eval approaches. This script helps understand how well different reviewers agree on answers and which metrics best capture inter-reviewer differences.

### `deepeval_results_analysis.py`
Analyzes and visualizes DeepEval evaluation results by merging results from multiple evaluation runs and creating comprehensive comparison plots. This script generates merged data files and various visualization plots for detailed analysis.

---

# DeepEval Benchmarking

The `deepeval_benchmarking.py` script provides comprehensive evaluation of LLM outputs against human reviewer (gold standard) answers using DeepEval's advanced metrics.

## Overview

This script evaluates LLM-generated answers against human reviewer answers using three key DeepEval metrics:
- **FaithfulnessMetric**: Measures how faithful the LLM answer is to the expected output
- **ContextualPrecisionMetric**: Measures precision in the context of the expected answer
- **ContextualRecallMetric**: Measures recall in the context of the expected answer

## Key Features

### 🚀 **Batch Processing with Incremental Saving**
- Processes test cases in configurable batches (default: 50)
- Saves results after each batch to prevent data loss
- Never lose progress again, even if interrupted

### 💰 **Cost-Effective Evaluation**
- **Default Model**: GPT-4o-mini (200x cheaper than GPT-4)
- **Configurable Models**: Easy switching between OpenAI models
- **Cost Display**: Real-time cost information and recommendations

### 🛡️ **Robust Error Handling**
- **Retry Logic**: Configurable retry attempts per batch (default: 5)
- **Context Truncation**: Automatic handling of long context to avoid API limits
- **Graceful Degradation**: Continues processing even if some batches fail

### 📁 **Organized Output**
- **Timestamped Filenames**: No file overwriting, easy tracking
- **Dedicated Directory**: `llm_benchmarking/deepeval-results/`
- **Dual Format**: JSON (human reading) + JSONL (programmatic processing)

## Usage

### Basic Usage
```bash
# Evaluate all questions with default settings
python llm_benchmarking/deepeval_benchmarking.py

# Evaluate specific question type
python llm_benchmarking/deepeval_benchmarking.py --question bee_species

# Evaluate with custom batch size
python llm_benchmarking/deepeval_benchmarking.py --batch-size 100
```

### Command Line Options
- `--question, -q`: Filter by question type
  - Choices: `bee_species`, `pesticides`, `additional_stressors`, `experimental_methodology`, `significance`, `future_research`, `limitations`
- `--limit, -l`: Maximum number of test cases to process
- `--batch-size, -b`: Number of test cases per batch (default: 50)
- `--max-retries, -r`: Maximum retries per batch (default: 5)
- `--model, -m`: OpenAI model for evaluation
  - Choices: `gpt-4o-mini` (default), `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`

### Examples
```bash
# Process all bee_species questions with large batches
python llm_benchmarking/deepeval_benchmarking.py --question bee_species --batch-size 198

# Process all questions with custom retry settings
python llm_benchmarking/deepeval_benchmarking.py --batch-size 100 --max-retries 3

# Use GPT-4o for better performance (higher cost)
python llm_benchmarking/deepeval_benchmarking.py --model gpt-4o

# Test run with limited cases
python llm_benchmarking/deepeval_benchmarking.py --limit 50 --batch-size 25
```

## Output Files

### Directory Structure
```
llm_benchmarking/deepeval-results/
├── deepeval_results_bee_species_20250820_103304.json      # Main results (JSON)
├── deepeval_results_bee_species_20250820_103304.jsonl     # Results (JSONL format)
└── ... (timestamped files for each run)
```

### File Formats
- **`.json`**: Human-readable format with proper structure and indentation
- **`.jsonl`**: Line-by-line format for easy processing, analysis, and database imports

### File Naming Convention
- **Format**: `deepeval_results_{question_type}_{timestamp}.{ext}`
- **Example**: `deepeval_results_bee_species_20250820_103304.json`
- **Benefits**: No overwriting, easy tracking, organized by question type

## Cost Information

### Model Cost Comparison
| Model | Input (per 1K tokens) | Output (per 1K tokens) | Recommendation |
|-------|----------------------|----------------------|----------------|
| **gpt-4o-mini** | $0.00015 | $0.0006 | **Most cost-effective** |
| gpt-4o | $0.0025 | $0.01 | Balanced performance/cost |
| gpt-4-turbo | $0.01 | $0.03 | Higher cost, better performance |
| gpt-3.5-turbo | $0.0005 | $0.0015 | Good cost, lower performance |

### Cost Savings
- **GPT-4o-mini vs GPT-4**: 200x cheaper for input, 100x cheaper for output
- **Recommended**: Use GPT-4o-mini for evaluation (default)

## Evaluation Metrics

### FaithfulnessMetric
- **Purpose**: Measures if the LLM answer is faithful to the expected output
- **Score Range**: 0.0 - 1.0 (higher = more faithful)
- **Use Case**: Detecting hallucinations or extra information

### ContextualPrecisionMetric
- **Purpose**: Measures precision of retrieved context relative to expected answer
- **Score Range**: 0.0 - 1.0 (higher = more precise)
- **Use Case**: Assessing relevance of provided context

### ContextualRecallMetric
- **Purpose**: Measures recall of retrieved context relative to expected answer
- **Score Range**: 0.0 - 1.0 (higher = better recall)
- **Use Case**: Ensuring complete coverage of expected information

## Batch Processing Strategy

### Recommended Batch Sizes
- **Small Datasets** (< 100 cases): Batch size 25-50
- **Medium Datasets** (100-500 cases): Batch size 50-100
- **Large Datasets** (> 500 cases): Batch size 100-200

### Safety Features
- **Incremental Saving**: Results saved after each batch
- **Retry Logic**: Failed batches retry up to max_retries times
- **Context Truncation**: Prevents API token limit issues
- **Progress Tracking**: Real-time progress indicators

## Error Handling

### Automatic Recovery
- **API Failures**: Automatic retry with exponential backoff
- **Context Limits**: Automatic truncation to stay under API limits
- **Batch Failures**: Skip failed batches, continue with next
- **Data Loss Prevention**: Incremental saving ensures progress is never lost

### Monitoring
- **Real-time Progress**: Shows current batch and overall progress
- **Error Logging**: Detailed error messages for debugging
- **Success Tracking**: Counts successful vs. failed test cases
- **Cost Tracking**: Shows evaluation costs per metric

## Performance Optimization

### Context Management
- **Automatic Truncation**: Limits context to 10 chunks, max 1,000 chars each
- **Token Limit Prevention**: Stops at 8,000 total chars to avoid API limits
- **Smart Fallbacks**: Uses context as retrieval_context if missing

### Batch Optimization
- **Concurrent Processing**: Configurable concurrency (default: 10)
- **Memory Management**: Processes batches sequentially to manage memory
- **API Rate Limiting**: Built-in throttling to avoid overwhelming APIs

## Use Cases

### Research Evaluation
- **LLM Performance Assessment**: Compare LLM outputs against human experts
- **Quality Metrics**: Quantify answer faithfulness and completeness
- **Cost Analysis**: Track evaluation costs across different models

### Production Systems
- **Model Selection**: Evaluate different LLM models for specific tasks
- **Prompt Engineering**: Test prompt variations and measure effectiveness
- **Quality Assurance**: Automated quality checking of LLM outputs

### Academic Research
- **Benchmarking Studies**: Standardized evaluation across research groups
- **Meta-analyses**: Compare results across different studies and datasets
- **Reproducibility**: Timestamped results for research validation

## Dependencies

- **Required**: `deepeval`, `openai`
- **Built-in**: `json`, `os`, `datetime`, `argparse`
- **Environment**: `OPENAI_API_KEY`, `CREATIVE_AI_API_KEY`

## Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (for additional features)
CREATIVE_AI_API_KEY=your_creative_ai_key_here
```

### Default Settings
- **Model**: GPT-4o-mini
- **Batch Size**: 50 test cases
- **Max Retries**: 5 per batch
- **Output Directory**: `llm_benchmarking/deepeval-results/`

## Troubleshooting

### Common Issues
1. **API Key Errors**: Ensure `OPENAI_API_KEY` is set in environment
2. **Memory Issues**: Reduce batch size for large datasets
3. **API Limits**: Increase retry count or reduce concurrency
4. **Context Errors**: Script automatically handles long context

### Performance Tips
- **Start Small**: Begin with small batch sizes and increase gradually
- **Monitor Costs**: Use GPT-4o-mini for cost-effective evaluation
- **Check Progress**: Monitor incremental saves to ensure no data loss
- **Review Logs**: Check console output for detailed progress information

---

# Data Merging

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

---

# Reviewer Dataset Generation

The `reviewer_dataset_generation.py` script generates test datasets that compare answers from two different reviewers for the same questions, enabling analysis of inter-reviewer agreement and consensus patterns.

## Overview

This script processes the merged data files to create a specialized dataset for reviewer comparison analysis. It extracts cases where both reviewers have provided answers and creates a structured format for evaluation.

## Key Features

### 🔍 **Dual Reviewer Focus**
- Only processes entries with both `answer_rev1` and `answer_rev2` fields
- Skips papers with only one reviewer
- Ensures data quality by filtering empty answers

### 📊 **Structured Output**
- **JSON format**: Complete dataset with full metadata
- **CSV format**: Tabular format for easy analysis
- **Metadata preservation**: Includes both reviewers' ratings and identifiers

### 🎯 **Use Cases**
- Evaluating inter-reviewer agreement
- Training models to identify consensus vs. disagreement
- Understanding how different reviewers interpret the same questions
- Quality assessment of reviewer 2 relative to reviewer 1

## Usage

### Basic Usage
```bash
# Generate reviewer comparison dataset
python llm_benchmarking/reviewer_dataset_generation.py
```

### Prerequisites
- `METABEEAI_DATA_DIR` environment variable set
- `final_merged_data/*_merged.json` files available
- `llm_questions.txt` file with question definitions

## Output Files

### Generated Files
- **`rev_test_dataset.json`**: Complete dataset in JSON format
- **`rev_test_dataset.csv`**: Dataset in CSV format for analysis
- **Output Location**: `llm_benchmarking/test-datasets/` folder

### Data Structure
```json
{
  "id": "594",
  "input": "What species of bee(s) were tested?",
  "expected_output": "Honey bees (*Apis mellifera*), specifically the Carniolan honey bee (*Apis mellifera carnica*), were tested in the study.",
  "actual_outputs": "The bee species tested was Apis mellifera. The subspecies was Apis mellifera carnica.",
  "metadata": {
    "paper_id": "594",
    "question_id": "bee_species",
    "rev1": "AJ",
    "rev2": "LH",
    "rev1_rating": 10,
    "rev2_rating": 10
  }
}
```

## Statistics Provided

The script generates comprehensive statistics including:
- Total entries generated
- Entries per question type
- Entries per reviewer pair
- Rating distribution analysis

---

# Reviewer Comparison Evaluation

The `deepeval_reviewers.py` script evaluates the reviewer comparison dataset using context-free metrics to assess inter-reviewer agreement and identify patterns in consensus vs. disagreement.

## Overview

This script uses a hybrid approach combining traditional DeepEval metrics with G-Eval metrics to provide comprehensive evaluation of reviewer agreement without requiring paper context.

## Metrics Used

### 🔍 **Context-Free Evaluation**
- **FaithfulnessMetric**: Measures how faithful reviewer 2 answers are to reviewer 1
- **G-Eval Correctness**: Strict evaluation of reviewer 2 accuracy against reviewer 1
- **G-Eval Completeness**: Assessment of reviewer 2 coverage of reviewer 1 key points
- **G-Eval Accuracy**: Evaluation of reviewer 2 information accuracy vs reviewer 1

### ⚡ **Why Context-Free?**
- **FaithfulnessMetric**: Only compares actual vs expected output
- **G-Eval Metrics**: Use only `ACTUAL_OUTPUT` and `EXPECTED_OUTPUT` parameters
- **No Context Required**: Avoids need for full paper text or retrieval context

## Key Features

### 🚀 **Batch Processing**
- Configurable batch sizes (default: 50)
- Incremental saving to prevent data loss
- Retry logic with exponential backoff

### 📊 **Comprehensive Analysis**
- Metric scores for each evaluation type
- Reviewer agreement analysis by reviewer pair
- Success rate tracking
- Detailed results in JSON and JSONL formats

### 💰 **Cost Optimization**
- Uses GPT-4o by default for best quality
- Configurable model selection
- Cost information and recommendations

## Usage

### Basic Usage
```bash
# Evaluate all reviewer comparisons
python llm_benchmarking/deepeval_reviewers.py

# Evaluate specific question type
python llm_benchmarking/deepeval_reviewers.py --question bee_species

# Evaluate with custom settings
python llm_benchmarking/deepeval_reviewers.py --question pesticides --batch-size 25 --model gpt-4o
```

### Command Line Options
- `--question, -q`: Filter by question type
- `--limit, -l`: Maximum number of test cases to process
- `--batch-size, -b`: Number of test cases per batch (default: 50)
- `--max-retries, -r`: Maximum retries per batch (default: 5)
- `--model, -m`: OpenAI model for evaluation (default: gpt-4o)

## Output Files

### Directory Structure
```
llm_benchmarking/deepeval-results/
├── deepeval_reviewer_results_bee_species_20250120_143022.json
├── deepeval_reviewer_results_bee_species_20250120_143022.jsonl
└── ... (timestamped files for each run)
```

### Analysis Results
- **Metric Scores**: Average scores for each evaluation metric
- **Reviewer Agreement**: Agreement scores between different reviewer pairs
- **Success Rates**: Percentage of successful evaluations
- **Detailed Results**: Full evaluation data with metadata

## Benefits for Research

### 🔬 **Inter-Reviewer Analysis**
- Understand how well different reviewers agree
- Identify systematic differences in interpretation
- Assess consistency across reviewer pairs

### 📈 **Quality Assessment**
- Evaluate reviewer 2 performance relative to reviewer 1
- Identify areas of strong agreement vs. disagreement
- Support quality improvement initiatives

### 🤖 **Model Training**
- Train models to identify consensus patterns
- Develop disagreement detection systems
- Support automated quality assessment

## Dependencies

- **Required**: `deepeval`, `openai`, `python-dotenv`
- **Built-in**: `json`, `os`, `argparse`, `datetime`

---

# DeepEval Results Analysis

The `deepeval_results_analysis.py` script provides comprehensive analysis and visualization of DeepEval evaluation results by merging multiple evaluation runs and creating detailed comparison plots.

## Overview

This script consolidates results from various DeepEval evaluation runs and generates:
- Merged data files without context fields for easier analysis
- Comparison plots between LLM vs reviewer and reviewer vs reviewer data
- Individual metric plots for detailed analysis by question type

## Key Features

### 🔄 **Data Consolidation**
- Automatically identifies and loads all DeepEval result files
- Separates LLM vs reviewer and reviewer vs reviewer data
- Removes context and retrieval_context fields for cleaner analysis
- Creates standardized merged data files

### 📊 **Comprehensive Visualization**
- **Average Scores Plot**: Bar plot comparing all metrics across data types
- **Individual Metric Plots**: Detailed analysis for each metric by question type
- **Standard Error Bars**: Statistical significance indicators
- **Binary vs Continuous**: Special handling for Correctness (binary) vs other metrics

### 📁 **Organized Output**
- **Merged Data**: JSON and JSONL files in `deepeval-analyses/merged-data/`
- **Plots**: High-resolution PNG files in `deepeval-analyses/`
- **Summary**: Comprehensive analysis summary in JSON format

## Usage

### Basic Usage
```bash
# Run complete analysis pipeline
python llm_benchmarking/deepeval_results_analysis.py
```

### What It Does
1. **Loads Results**: Finds all DeepEval result files in `deepeval-results/`
2. **Separates Data**: Distinguishes between LLM and reviewer comparison results
3. **Cleans Data**: Removes context fields and standardizes structure
4. **Creates Plots**: Generates comparison and individual metric visualizations
5. **Saves Output**: Creates organized output files and directories

## Output Structure

### Generated Files
```
llm_benchmarking/deepeval-analyses/
├── merged-data/
│   ├── merged_llm_comparison_results.json
│   ├── merged_llm_comparison_results.jsonl
│   ├── merged_reviewer_comparison_results.json
│   └── merged_reviewer_comparison_results.jsonl
├── average_scores_comparison.png
├── FaithfulnessMetric_by_question_type.png
├── ContextualPrecisionMetric_by_question_type.png
├── ContextualRecallMetric_by_question_type.png
├── Correctness_by_question_type.png
├── Completeness_by_question_type.png
├── Accuracy_by_question_type.png
└── analysis_summary.json
```

### Plot Types

#### **Average Scores Comparison**
- Compares all metrics across LLM vs reviewer and reviewer vs reviewer data
- Shows standard error bars for statistical significance
- Distinguishes between binary (Correctness) and continuous metrics

#### **Individual Metric Plots**
- One plot per metric showing performance by question type
- Side-by-side bars for different data types
- Error bars for statistical confidence

## Data Handling

### **Automatic Detection**
- **LLM vs Reviewer**: Files without "reviewer" in filename
- **Reviewer vs Reviewer**: Files with "reviewer" in filename

### **Context Removal**
- Removes `context` and `retrieval_context` fields
- Preserves all metric data and metadata
- Maintains data integrity for analysis

### **Question Type Extraction**
- Extracts from metadata when available
- Falls back to input text analysis
- Handles all 7 question types automatically

## Benefits for Research

### 🔬 **Comprehensive Analysis**
- Compare performance across different evaluation approaches
- Identify strengths and weaknesses of different metrics
- Understand patterns in LLM vs human performance

### 📈 **Visual Insights**
- Clear visual representation of complex evaluation data
- Easy identification of performance trends
- Support for presentations and publications

### 🔄 **Data Management**
- Consolidated data files for further analysis
- Clean, standardized format for external tools
- Historical tracking of evaluation results

## Dependencies

- **Required**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Built-in**: `json`, `os`, `glob`, `pathlib`, `typing`

## Status

**✅ Successfully Tested**: The script has been tested with real DeepEval results and successfully:
- Processed 1,658 evaluation entries
- Analyzed 6 different metrics across 6 question types
- Generated comprehensive visualizations and merged data files
- Created organized output structure in `deepeval-analyses/`
