# LLM Benchmarking and Evaluation Pipeline

This folder contains scripts for comprehensive benchmarking and evaluation of LLM-generated answers against human reviewer annotations. The pipeline supports multiple comparison types and provides detailed metric analysis.

## Overview

The benchmarking pipeline evaluates three types of comparisons:
1. **LLM v1 vs Reviewer 1** - Original LLM baseline performance
2. **LLM v2 vs Reviewer 3** - Improved LLM version performance  
3. **Reviewer 1 vs Reviewer 2** - Inter-reviewer agreement analysis

---

## Quick Start

### Prerequisites

1. **Environment Setup**:
```bash
# Activate your virtual environment
source ../venv/bin/activate  # On Mac/Linux
```

2. **API Keys**: Configure in `.env` file (project root):
```bash
OPENAI_API_KEY=your_openai_api_key_here
METABEEAI_DATA_DIR=/path/to/your/data
```

3. **Required Data Structure**:
```
papers/
├── 001/
│   └── answers.json  # LLM-generated answers with chunk_ids
final_merged_data/
├── bee_species_merged.json
├── pesticides_merged.json
└── ... (other question types)
```

---

### Basic Usage - Main Runner Script

The `run_benchmarking.py` script orchestrates the complete pipeline:

```bash
# Run LLM v1 benchmarking (all questions)
python run_benchmarking.py --type llmv1

# Run LLM v2 benchmarking for specific question
python run_benchmarking.py --type llmv2 --question bee_species

# Run reviewer comparison benchmarking
python run_benchmarking.py --type rev

# Run all benchmarking types
python run_benchmarking.py --type all

# Skip dataset generation (if already done)
python run_benchmarking.py --type llmv1 --skip-dataset

# Run only analysis (after benchmarking is done)
python run_benchmarking.py --analyze-only
```

**Command-line options**:
- `--type {llmv1,llmv2,rev,all}`: Type of benchmarking to run (required unless --analyze-only)
- `--question QUESTION`: Specific question type to evaluate (optional, default: all)
- `--skip-dataset`: Skip dataset generation step (use existing datasets)
- `--skip-geval`: Skip GEval evaluation (only for llmv1)
- `--faithfulness-only`: Run only faithfulness metric (only for rev type)
- `--analyze-only`: Only run analysis and visualization
- `--no-analysis`: Skip analysis step after benchmarking

---

## Complete Workflow

### STEP 1: Data Preparation

Before benchmarking, you need merged data combining LLM and reviewer answers:

#### 1A. Merge LLM v1 and Reviewer Answers
```bash
python merge_answers.py \
  --reviewer-db /path/to/reviewer/database \
  --data-dir /path/to/papers
```

**Input**:
- Reviewer database with structure: `reviewer_db/{reviewer_initials}/{paper_id}/answers_extended.json`
- LLM answers in: `data_dir/{paper_id}/answers.json`

**Output**: `final_merged_data/*.json` files
- Each file contains merged LLM and reviewer answers for one question type
- Structure: `{paper_id: {answer_llm, answer_rev1, rev1_rating, ...}}`

#### 1B. Add LLM v2 Answers (Optional)
```bash
python merge_llm_v2.py
```

**Input**: 
- Existing `final_merged_data/*.json` files
- New LLM v2 answers in papers directory

**Output**: Updates `final_merged_data/*.json` with `answer_llmv2` fields

---

### STEP 2: Dataset Generation

Generate test datasets for evaluation:

#### 2A. LLM vs Reviewer Test Dataset
```bash
python test_dataset_generation.py
```

**What it does**:
- Loads merged data from `final_merged_data/`
- Pairs LLM answers (actual_output) with reviewer answers (expected_output)
- Retrieves text chunks from papers as context
- Creates evaluation-ready test cases

**Output**: `test-datasets/test_dataset.json` and `test_dataset.csv`

**Dataset structure**:
```json
{
  "id": "729",
  "input": "What species of bee(s) were tested?",
  "actual_outputs": "LLM generated answer...",
  "expected_output": "Reviewer gold standard answer...",
  "context": ["chunk 1 text...", "chunk 2 text..."],
  "retrieval_context": ["chunk 1 text...", "chunk 2 text..."],
  "metadata": {
    "paper_id": "729",
    "question_id": "bee_species",
    "rev1_rating": 10,
    "rev1": "AB"
  }
}
```

#### 2B. Reviewer Comparison Test Dataset
```bash
python reviewer_dataset_generation.py
```

**What it does**:
- Creates dataset comparing Reviewer 1 vs Reviewer 2
- Only includes papers with both reviewers
- Uses Rev1 as expected_output, Rev2 as actual_outputs

**Output**: `test-datasets/rev_test_dataset.json` and `rev_test_dataset.csv`

---

### STEP 3: Run Benchmarking Evaluations

#### Using Main Runner (Recommended)

```bash
# Run complete LLM v1 pipeline
python run_benchmarking.py --type llmv1

# Run complete LLM v2 pipeline
python run_benchmarking.py --type llmv2

# Run complete reviewer comparison pipeline
python run_benchmarking.py --type rev
```

#### Running Individual Scripts

##### 3A. DeepEval Standard Metrics (LLM v1)
```bash
# All questions
python deepeval_benchmarking.py

# Specific question
python deepeval_benchmarking.py --question bee_species

# Custom settings
python deepeval_benchmarking.py --question pesticides --batch-size 25
```

**Metrics evaluated**:
- **FaithfulnessMetric**: Measures how faithful actual output is to expected output
- **ContextualPrecisionMetric**: Measures precision of context retrieval
- **ContextualRecallMetric**: Measures completeness of context coverage

**Output**: `deepeval-results/deepeval_results_{question}_{timestamp}.json/jsonl`

##### 3B. GEval Metrics (LLM v1)
```bash
python deepeval_GEval.py --question bee_species
```

**Metrics evaluated**:
- **Correctness**: Binary evaluation of factual accuracy
- **Completeness**: Coverage of key points
- **Accuracy**: Semantic alignment quality

**Output**: `deepeval-results/deepeval_GEval_results_{question}_{timestamp}.json/jsonl`

##### 3C. LLM v2 Evaluation
```bash
python deepeval_llmv2.py --question bee_species
```

**What it does**:
- Evaluates LLM v2 answers against Reviewer 3
- Uses same metrics as standard DeepEval
- Automatically loads LLM v2 data from merged files

**Output**: `deepeval-results/deepeval_llmv2_vs_rev3_results_{question}_{timestamp}.json/jsonl`

##### 3D. Reviewer Comparison Evaluation
```bash
# Full evaluation (GEval + Faithfulness)
python deepeval_reviewers.py

# Faithfulness only (faster, cheaper)
python deepeval_reviewers.py --faithfulness-only --add-context

# Specific question
python deepeval_reviewers.py --question bee_species
```

**What it does**:
- Evaluates agreement between Reviewer 1 and Reviewer 2
- Uses context-free metrics (no paper context needed for GEval)
- Optional: Add FaithfulnessMetric with context for deeper analysis

**Output**: `deepeval-results/deepeval_reviewer_results_{question}_{timestamp}.json/jsonl`

---

### STEP 4: Analysis and Visualization

#### Using Main Runner

```bash
# Run analysis after benchmarking
# (automatically runs unless --no-analysis specified)
python run_benchmarking.py --analyze-only
```

#### Running Individual Scripts

##### 4A. Generate Comprehensive Summary
```bash
python analyze_deepeval_results_improved.py
```

**What it does**:
- Scans all `.jsonl` files in `deepeval-results/`
- Identifies comparison types (llm_vs_rev1, llmv2_vs_rev3, reviewer_comparison, GEval)
- Extracts question types from metadata or filenames
- Calculates statistics: mean, count, std, standard error

**Output**: `deepeval-results/summary_all_results_improved.csv`

**CSV structure**:
```csv
Comparison,Question,Metric,Mean,Count,Std,SE,Data_Type
llm_vs_rev1,bee_species,FaithfulnessMetric,0.85,198,0.12,0.009,llm_vs_rev1
llmv2_vs_rev3,bee_species,FaithfulnessMetric,0.92,198,0.09,0.006,llmv2_vs_rev3
reviewer_comparison,bee_species,FaithfulnessMetric,0.95,98,0.07,0.007,reviewer_comparison
```

##### 4B. Create Merged Data and Plots
```bash
python deepeval_results_analysis.py
```

**What it does**:
- Merges all evaluation results into consolidated files
- Removes context fields for cleaner analysis
- Creates individual metric plots by question type
- Generates average scores comparison plot

**Output** in `deepeval-analyses/`:
- `merged-data/merged_llm_comparison_results.json/jsonl`
- `merged-data/merged_reviewer_comparison_results.json/jsonl`
- `average_scores_comparison.png` - Bar plot comparing all metrics
- `{Metric}_by_question_type.png` - Individual plots per metric
- `analysis_summary.json` - Statistical summary

##### 4C. Create Comparison Plots
```bash
python create_comparison_plots.py
```

**What it does**:
- Creates grid and horizontal comparison plots
- Compares performance across question types and comparison types
- Generates summary statistics table

**Output** in `comparison_plots/`:
- `{metric}_grid.pdf` - Grid layout comparison
- `{metric}_horizontal.pdf` - Horizontal bar chart
- `overall_average_performance_grid.pdf` - Summary across all metrics
- `summary_statistics.csv` - Detailed statistics table

---

### STEP 5: Additional Analyses (Optional)

#### Reviewer Rating Analysis
```bash
python reviewer_rating.py
```

**What it does**:
- Analyzes reviewer ratings from merged data
- Calculates average ratings by question type
- Measures inter-reviewer agreement
- Compares individual reviewer performance

**Output** in `analyses/`:
- `avg-rev-ratings.png` - Average ratings by question
- `avg-rev-agreement.png` - Inter-reviewer agreement scores
- `avg-score-per-reviewer.png` - Individual reviewer comparison

#### Edge Case Analysis
```bash
python edge_cases.py
```

**What it does**:
- Identifies low-scoring evaluation cases
- Extracts papers with poor LLM performance
- Groups edge cases by question type and issue

**Output** in `edge-cases/`:
- `llm/*.json` - Low-scoring LLM cases by question
- `reviewer/*.json` - Low-scoring reviewer cases
- `edge-case-report.md` - Detailed analysis report
- `summary-report.json` - Statistical summary

---

## Understanding the Metrics

### DeepEval Standard Metrics

#### FaithfulnessMetric
- **What it measures**: How well the actual output aligns with expected output and context
- **Score range**: 0.0 - 1.0 (higher = more faithful)
- **Use case**: Detecting hallucinations and context violations
- **Example**: If LLM says "50ppb" but paper says "30ppb", score is low

#### ContextualPrecisionMetric
- **What it measures**: Relevance and precision of retrieved context
- **Score range**: 0.0 - 1.0 (higher = more precise)
- **Use case**: Evaluating context selection quality
- **Example**: High score if context contains specific details; low if too general

#### ContextualRecallMetric
- **What it measures**: Completeness of context coverage
- **Score range**: 0.0 - 1.0 (higher = more complete)
- **Use case**: Ensuring all necessary information is retrieved
- **Example**: High score if context includes all species, doses, methods

### GEval Metrics

#### Correctness
- **What it measures**: Binary evaluation of factual accuracy
- **Score range**: 0.0 or 1.0 (binary)
- **Use case**: Identifying clear factual errors
- **Example**: Either correct or incorrect, no partial credit

#### Completeness
- **What it measures**: Coverage of all key points
- **Score range**: 0.0 - 1.0 (continuous)
- **Use case**: Measuring answer thoroughness
- **Example**: Partial credit for partially complete answers

#### Accuracy
- **What it measures**: Semantic alignment and quality
- **Score range**: 0.0 - 1.0 (continuous)
- **Use case**: Nuanced assessment of answer quality
- **Example**: Captures differences in detail level and emphasis

---

## Output File Organization

```
llm_benchmarking/
├── test-datasets/
│   ├── test_dataset.json           # LLM vs Reviewer dataset
│   ├── test_dataset.csv
│   ├── rev_test_dataset.json       # Reviewer vs Reviewer dataset
│   └── rev_test_dataset.csv
├── deepeval-results/
│   ├── deepeval_results_{question}_{timestamp}.json/jsonl           # Standard metrics
│   ├── deepeval_GEval_results_{question}_{timestamp}.json/jsonl     # GEval metrics
│   ├── deepeval_llmv2_vs_rev3_results_{question}_{timestamp}.json/jsonl  # LLM v2
│   ├── deepeval_reviewer_results_{question}_{timestamp}.json/jsonl  # Reviewer comparison
│   └── summary_all_results_improved.csv                             # Comprehensive summary
├── deepeval-analyses/
│   ├── merged-data/
│   │   ├── merged_llm_comparison_results.json/jsonl
│   │   └── merged_reviewer_comparison_results.json/jsonl
│   ├── average_scores_comparison.png
│   ├── {Metric}_by_question_type.png
│   └── analysis_summary.json
├── comparison_plots/
│   ├── {metric}_grid.pdf
│   ├── {metric}_horizontal.pdf
│   ├── overall_average_performance_grid.pdf
│   └── summary_statistics.csv
├── edge-cases/
│   ├── llm/*.json
│   ├── reviewer/*.json
│   └── edge-case-report.md
└── final_merged_data/
    ├── bee_species_merged.json
    ├── pesticides_merged.json
    └── ... (other question types)
```

---

## Cost Optimization

### Model Selection

| Model | Cost | Speed | Quality | Recommendation |
|-------|------|-------|---------|----------------|
| **gpt-4o-mini** | Very Low | Fast | Good | Recommended for most evaluations |
| gpt-4o | Medium | Medium | Excellent | Use for critical evaluations |
| gpt-4-turbo | High | Fast | Excellent | High-budget projects |
| gpt-3.5-turbo | Very Low | Very Fast | Fair | Quick tests only |

### Cost Reduction Tips

1. **Use gpt-4o-mini** for initial evaluation runs (default)
2. **Start with small batches** to estimate costs
3. **Filter by question type** for targeted evaluation
4. **Skip dataset generation** if datasets already exist
5. **Use --faithfulness-only** for reviewer comparisons (cheaper than full GEval)

### Example Cost Calculation

For 200 test cases with gpt-4o-mini:
- Input tokens: ~2,000 per case × 200 = 400,000 tokens ≈ $0.06
- Output tokens: ~500 per case × 200 = 100,000 tokens ≈ $0.06
- **Total**: ~$0.12 per metric
- **Full evaluation** (3 metrics): ~$0.36

---

## Troubleshooting

### "METABEEAI_DATA_DIR not set"
- **Fix**: Add to `.env` file: `METABEEAI_DATA_DIR=/path/to/data`

### "No test cases found"
- **Cause**: Missing or empty test dataset
- **Fix**: Run `test_dataset_generation.py` or `reviewer_dataset_generation.py`

### "API rate limit exceeded"
- **Cause**: Too many concurrent requests
- **Fix**: Reduce batch size: `--batch-size 10`

### "Context too long" errors
- **Cause**: Papers with very long context
- **Fix**: Reduce batch size or use `--max-context-length 40000`

### Low faithfulness scores
- **Expected**: Review edge cases in `edge-cases/` folder
- **Action**: Check if LLM answers differ significantly from reviewers

### Missing merged data files
- **Cause**: Haven't run merge_answers.py
- **Fix**: Run STEP 1 data preparation first

---

## Advanced Usage

### Custom Batch Processing

```bash
# Small batches for long papers
python deepeval_benchmarking.py --question experimental_methodology --batch-size 10

# Large batches for short papers
python deepeval_benchmarking.py --question bee_species --batch-size 100
```

### Retry Configuration

```bash
# Increase retries for unstable connections
python deepeval_benchmarking.py --max-retries 10
```

### Partial Processing

```bash
# Process only first 50 test cases (testing)
python deepeval_benchmarking.py --question bee_species --limit 50

# Skip already-completed steps
python run_benchmarking.py --type llmv1 --skip-dataset --skip-geval
```

### Comparison Analysis

```bash
# Generate only analysis, skip benchmarking
python run_benchmarking.py --analyze-only

# Run benchmarking without analysis
python run_benchmarking.py --type llmv1 --no-analysis
```

---

## Dependencies

Core dependencies (install via `pip install -r ../requirements.txt`):
- `deepeval` - Evaluation framework
- `openai` - OpenAI API client
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `python-dotenv` - Environment variable management

---

## Next Steps After Benchmarking

1. **Review Summary CSV**: Check `deepeval-results/summary_all_results_improved.csv`
2. **Examine Plots**: Review visualizations in `deepeval-analyses/` and `comparison_plots/`
3. **Analyze Edge Cases**: Investigate low-scoring cases in `edge-cases/`
4. **Compare Versions**: Look at LLM v1 vs LLM v2 improvements
5. **Check Inter-Reviewer Agreement**: Review reviewer comparison results
6. **Iterate on Prompts**: Use insights to improve LLM prompts in `../metabeeai_llm/questions.yml`

---

## Related Documentation

- **LLM Pipeline**: See `../metabeeai_llm/README.md` for question answering pipeline
- **PDF Processing**: See `../process_pdfs/README.md` for data preparation
- **Configuration**: See `../config.py` for centralized settings

---

**Last Updated**: October 2025

