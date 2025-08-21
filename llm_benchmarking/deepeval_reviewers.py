#!/usr/bin/env python3
"""
DeepEval Reviewers Comparison Evaluation

This script evaluates the reviewer comparison dataset (rev_test_dataset.json) using
context-free metrics from both traditional DeepEval and G-Eval approaches.

Metrics used:
- FaithfulnessMetric: Measures how faithful reviewer 2 answers are to reviewer 1 answers
- G-Eval Correctness: Strict evaluation of reviewer 2 accuracy against reviewer 1
- G-Eval Completeness: Assessment of reviewer 2 coverage of reviewer 1 key points
- G-Eval Accuracy: Evaluation of reviewer 2 information accuracy vs reviewer 1

This evaluation helps understand:
- How well different reviewers agree on answers
- Which metrics best capture inter-reviewer differences
- Patterns in reviewer consensus vs. disagreement
"""

import json
import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Evaluate reviewer comparison dataset with DeepEval')
parser.add_argument('--question', '-q', 
                   choices=['bee_species', 'pesticides', 'additional_stressors', 'experimental_methodology', 'significance', 'future_research', 'limitations'],
                   help='Question type to filter by (optional - if not specified, processes all questions)')
parser.add_argument('--limit', '-l', type=int, 
                   help='Maximum number of test cases to process (optional)')
parser.add_argument('--batch-size', '-b', type=int, default=50,
                   help='Number of test cases to process per batch (default: 50)')
parser.add_argument('--max-retries', '-r', type=int, default=5,
                   help='Maximum retries per batch (default: 5)')
parser.add_argument('--model', '-m', type=str, default='gpt-4o',
                   choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo'],
                   help='OpenAI model to use for evaluation (default: gpt-4o for best quality)')
parser.add_argument('--add-context', action='store_true',
                   help='Add paper context to test cases (optional, for metrics that might benefit from it)')
parser.add_argument('--faithfulness-only', action='store_true',
                   help='Run only FaithfulnessMetric evaluation (requires --add-context)')

args = parser.parse_args()

# Set API keys from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
creative_ai_key = os.getenv("CREATIVE_AI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")
if not creative_ai_key:
    raise ValueError("CREATIVE_AI_API_KEY not found in .env file")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["CONFIDENT_API_KEY"] = creative_ai_key
print("✅ OpenAI API key loaded from .env file")
print("✅ Creative AI API key loaded from .env file")

from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig, DisplayConfig
from deepeval.models import GPTModel

# Load the reviewer comparison dataset
with open("llm_benchmarking/test-datasets/rev_test_dataset.json", "r") as f:
    data = json.load(f)

# Load the LLM dataset to get context if needed
llm_context_data = {}
if args.add_context:
    print("🔄 Loading LLM dataset to extract paper context...")
    try:
        with open("llm_benchmarking/test-datasets/test_dataset.json", "r") as f:
            llm_data = json.load(f)
            # Create a mapping from paper_id to context
            for entry in llm_data:
                if "id" in entry and "context" in entry:
                    llm_context_data[entry["id"]] = entry["context"]
        print(f"✅ Loaded context for {len(llm_context_data)} papers")
    except Exception as e:
        print(f"⚠️  Warning: Could not load LLM dataset for context: {e}")
        print("   Continuing without context (FaithfulnessMetric will not work)")
        args.add_context = False

# Filter by question type (optional)
print(f"Original dataset: {len(data)} test cases")

if args.question:
    filtered_data = [entry for entry in data if entry["metadata"]["question_id"] == args.question]
    print(f"Filtered by '{args.question}': {len(filtered_data)} test cases")
else:
    filtered_data = data
    print("Processing all question types")

# Apply limit if specified
if args.limit and len(filtered_data) > args.limit:
    filtered_data = filtered_data[:args.limit]
    print(f"Limited to first {args.limit} test cases")

# Create the dataset
dataset = EvaluationDataset()

# Add test cases to the dataset
skipped_count = 0
for i, entry in enumerate(filtered_data):
    if i % 10 == 0:  # Progress indicator
        print(f"Processing test case {i+1}/{len(filtered_data)}")
    
    # Check for required fields and skip if missing
    required_fields = ["input", "actual_outputs", "expected_output"]
    missing_fields = [field for field in required_fields if not entry.get(field)]
    
    if missing_fields:
        print(f"⚠️  Skipping test case {i+1}: Missing fields {missing_fields}")
        skipped_count += 1
        continue
    
    try:
        # Create LLMTestCase object with proper identifiers
        test_case_kwargs = {
            "input": entry["input"],
            "actual_output": entry["actual_outputs"],  # Reviewer 2 answer
            "expected_output": entry["expected_output"],  # Reviewer 1 answer (gold standard)
            "name": f"reviewer_comparison_{entry['id']}_case_{i}",  # Unique name for each test case
            "additional_metadata": {
                "paper_id": entry["id"],
                "question_id": entry["metadata"]["question_id"],
                "rev1": entry["metadata"]["rev1"],
                "rev2": entry["metadata"]["rev2"],
                "rev1_rating": entry["metadata"]["rev1_rating"],
                "rev2_rating": entry["metadata"]["rev2_rating"]
            }
        }
        
        # Add context if requested and available
        if args.add_context and entry["id"] in llm_context_data:
            # DeepEval expects context to be a list of strings
            test_case_kwargs["context"] = llm_context_data[entry["id"]]
            # Also set retrieval_context to the same value for compatibility
            test_case_kwargs["retrieval_context"] = llm_context_data[entry["id"]]
            total_chars = sum(len(chunk) for chunk in llm_context_data[entry["id"]])
            print(f"    📄 Added context for paper {entry['id']} ({len(llm_context_data[entry['id']])} context chunks, {total_chars} chars)")
        
        test_case = LLMTestCase(**test_case_kwargs)
        
        # Set proper identifiers to avoid ID errors
        test_case._identifier = f"reviewer_comparison_{entry['id']}_case_{i}"
        test_case._dataset_id = f"reviewer_comparison_{args.question if args.question else 'all'}"
        
        # Add as test case
        dataset.add_test_case(test_case)
        
    except Exception as e:
        print(f"⚠️  Error creating test case {i+1}: {e}")
        skipped_count += 1
        continue

if skipped_count > 0:
    print(f"⚠️  Skipped {skipped_count} test cases due to missing data or errors")

print("Dataset created successfully!")
print(f"Dataset contains {len(filtered_data)} test cases")

# Configure the specified model (default: GPT-4o for best quality)
evaluation_model = GPTModel(model=args.model)

# Define metrics for reviewer comparison evaluation
metrics = []

# Check if faithfulness-only mode is requested
if args.faithfulness_only:
    if not args.add_context:
        print("❌ Error: --faithfulness-only requires --add-context")
        print("   FaithfulnessMetric needs paper context to function")
        sys.exit(1)
    
    # Only add FaithfulnessMetric
    metrics.append(FaithfulnessMetric(model=evaluation_model))
    print("✅ Running FaithfulnessMetric only (requires context)")
    
else:
    # Add FaithfulnessMetric if context is available
    if args.add_context:
        metrics.append(FaithfulnessMetric(model=evaluation_model))
        print("✅ Added FaithfulnessMetric (requires context)")
    
    # G-Eval metrics (context-free, always included unless faithfulness-only)
    geval_metrics = [
        GEval(
            name="Correctness",
            criteria="Correctness - determine if the reviewer 2 answer is correct according to the reviewer 1 answer.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            strict_mode=True
        ),
        GEval(
            name="Completeness",
            criteria="Completeness - assess if the reviewer 2 answer covers all the key points mentioned in the reviewer 1 answer.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            strict_mode=False
        ),
        GEval(
            name="Accuracy",
            criteria="Accuracy - evaluate if the reviewer 2 answer contains accurate information that aligns with the reviewer 1 answer.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            strict_mode=False
        )
    ]
    
    metrics.extend(geval_metrics)
    print(f"✅ Added {len(geval_metrics)} G-Eval metrics (context-free)")

print(f"💰 Using {args.model} for evaluation (best quality option)")

# Show cost comparison
cost_info = {
    'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006, 'description': 'Most cost-effective'},
    'gpt-4o': {'input': 0.0025, 'output': 0.01, 'description': 'Balanced performance/cost'},
    'gpt-4-turbo': {'input': 0.01, 'output': 0.03, 'description': 'Higher cost, better performance'},
    'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015, 'description': 'Good cost, lower performance'}
}

selected_cost = cost_info[args.model]
print(f"💰 Cost per 1K tokens: Input ${selected_cost['input']:.4f}, Output ${selected_cost['output']:.4f}")
if args.model == 'gpt-4o':
    print(f"💰 Using GPT-4o for best evaluation quality - recommended for reviewer comparison!")
else:
    print(f"💰 Recommendation: {selected_cost['description']}")

metric_type = "context-aware" if args.add_context else "context-free"
print(f"Evaluating with {len(metrics)} {metric_type} metrics:")
for i, metric in enumerate(metrics):
    if hasattr(metric, 'name'):
        print(f"  {i+1}. {metric.name}: {metric.criteria}")
    else:
        print(f"  {i+1}. {metric.__class__.__name__}: Measures faithfulness of reviewer 2 to reviewer 1")

# Initialize results tracking with timestamped filenames
import datetime
import os

# Create timestamp for unique filenames
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
question_type = args.question if args.question else "all_questions"

# Create results directory if it doesn't exist
results_dir = "llm_benchmarking/deepeval-results"
os.makedirs(results_dir, exist_ok=True)

# Generate unique filenames in the results directory
if args.faithfulness_only:
    results_file = f"{results_dir}/deepeval_reviewer_faithfulness_{question_type}_{timestamp}.json"
    results_jsonl_file = f"{results_dir}/deepeval_reviewer_faithfulness_{question_type}_{timestamp}.jsonl"
else:
    results_file = f"{results_dir}/deepeval_reviewer_results_{question_type}_{timestamp}.json"
    results_jsonl_file = f"{results_dir}/deepeval_reviewer_results_{question_type}_{timestamp}.jsonl"

print(f"📁 Output files will be saved in: {results_dir}/")
if args.faithfulness_only:
    print(f"📁 File prefix: deepeval_reviewer_faithfulness_{question_type}_{timestamp}")
else:
    print(f"📁 File prefix: deepeval_reviewer_*_{question_type}_{timestamp}")

# Function to save results incrementally
def save_results_incrementally(results_list, filename, jsonl_filename):
    """Save results incrementally to prevent data loss"""
    try:
        # Save as JSON
        with open(filename, "w") as f:
            json.dump(results_list, f, indent=2)
        
        # Save as JSONL
        with open(jsonl_filename, "w") as f:
            for result in results_list:
                f.write(json.dumps(result) + "\n")
        
        print(f"💾 Incremental save: {len(results_list)} results saved to {filename}")
        return True
    except Exception as e:
        print(f"⚠️  Incremental save failed: {e}")
        return False

# Function to process test cases in batches with incremental saving
def process_test_cases_in_batches(test_cases, batch_size=50, max_retries=5):
    """Process test cases in batches and save incrementally with retry limits"""
    total_cases = len(test_cases)
    processed_results = []
    
    print(f"🔄 Processing {total_cases} test cases in batches of {batch_size}")
    print(f"🔄 Maximum retries per batch: {max_retries}")
    
    for batch_start in range(0, total_cases, batch_size):
        batch_end = min(batch_start + batch_size, total_cases)
        batch_cases = test_cases[batch_start:batch_end]
        
        print(f"\n📦 Processing batch {batch_start//batch_size + 1}: cases {batch_start+1}-{batch_end}")
        
        # Retry logic for each batch
        batch_success = False
        retry_count = 0
        
        while not batch_success and retry_count < max_retries:
            try:
                if retry_count > 0:
                    print(f"🔄 Retry attempt {retry_count}/{max_retries} for batch {batch_start//batch_size + 1}")
                
                # Process this batch
                batch_results = evaluate(
                    test_cases=batch_cases,
                    metrics=metrics,
                    async_config=AsyncConfig(
                        run_async=True, 
                        throttle_value=1,  # Add small delay between requests
                        max_concurrent=5   # Lower concurrency for better stability
                    ),
                    error_config=ErrorConfig(
                        ignore_errors=False,
                        skip_on_missing_params=True  # Skip problematic cases instead of failing
                    ),
                    display_config=DisplayConfig(
                        show_indicator=True,
                        print_results=False,  # Reduce output noise
                        verbose_mode=False
                    )
                )
                
                # If we get here, the batch was successful
                batch_success = True
                
                # Extract results from this batch
                if hasattr(batch_results, 'test_results') and batch_results.test_results:
                    for i, r in enumerate(batch_results.test_results):
                        try:
                            result_data = {
                                "test_case_index": len(processed_results) + i,
                                "name": getattr(r, 'name', f"batch_{batch_start//batch_size + 1}_case_{i}"),
                                "input": getattr(r, 'input', None),
                                "actual_output": getattr(r, 'actual_output', None),
                                "expected_output": getattr(r, 'expected_output', None),
                                "success": getattr(r, 'success', None),
                                "additional_metadata": getattr(r, 'additional_metadata', None),
                                "metrics_data": []
                            }
                            
                            # Extract metrics data
                            if hasattr(r, 'metrics_data') and r.metrics_data:
                                for metric in r.metrics_data:
                                    metric_data = {
                                        "name": getattr(metric, 'name', 'Unknown'),
                                        "score": getattr(metric, 'score', None),
                                        "threshold": getattr(metric, 'threshold', None),
                                        "success": getattr(metric, 'success', None),
                                        "reason": getattr(metric, 'reason', None),
                                        "strict_mode": getattr(metric, 'strict_mode', None),
                                        "evaluation_model": getattr(metric, 'evaluation_model', None),
                                        "error": getattr(metric, 'error', None),
                                        "evaluation_cost": getattr(metric, 'evaluation_cost', None)
                                    }
                                    result_data["metrics_data"].append(metric_data)
                            
                            processed_results.append(result_data)
                            
                        except Exception as e:
                            print(f"⚠️  Error extracting result {i}: {e}")
                            continue
                    
                    print(f"✅ Batch {batch_start//batch_size + 1} completed: {len(batch_results.test_results)} results")
                    
                    # Save incrementally after each batch
                if save_results_incrementally(processed_results, results_file, results_jsonl_file):
                    print(f"💾 Progress saved: {len(processed_results)}/{total_cases} test cases completed")
                else:
                    print(f"⚠️  Failed to save progress for batch {batch_start//batch_size + 1}")
                
            except Exception as batch_error:
                retry_count += 1
                error_msg = str(batch_error)
                
                print(f"❌ Batch {batch_start//batch_size + 1} failed (attempt {retry_count}/{max_retries}): {batch_error}")
                
                if retry_count >= max_retries:
                    print(f"💀 Batch {batch_start//batch_size + 1} failed after {max_retries} attempts - skipping to next batch")
                    break
                else:
                    # Exponential backoff: wait longer between retries
                    wait_time = min(5 * (2 ** (retry_count - 1)), 30)  # Max 30 seconds
                    print(f"💡 Waiting {wait_time} seconds before retry...")
                    import time
                    time.sleep(wait_time)
                    continue
        
        if not batch_success:
            print(f"⚠️  Skipping batch {batch_start//batch_size + 1} due to repeated failures")
    
    return processed_results

# Run evaluation with batch processing and incremental saving
print(f"\n🚀 Starting reviewer comparison evaluation of {len(dataset.test_cases)} test cases...")
print("⏳ Processing in batches with incremental saving to prevent data loss...")

# Process test cases in batches with incremental saving
batch_size = args.batch_size  # Process test cases per batch
max_retries = args.max_retries  # Maximum retries per batch
evaluation_results = process_test_cases_in_batches(dataset.test_cases, batch_size, max_retries)

if evaluation_results:
    print(f"\n🎉 Reviewer comparison evaluation completed successfully!")
    print(f"📊 Total results processed: {len(evaluation_results)}")
    print(f"📁 Final results saved to: {results_file}")
    print(f"📁 Final results also saved to: {results_jsonl_file}")
    
    # Print summary statistics
    successful_cases = sum(1 for r in evaluation_results if r.get('success', False))
    print(f"✅ Successful test cases: {successful_cases}/{len(evaluation_results)}")
    
    # Calculate average scores for each metric
    metric_scores = {}
    for result in evaluation_results:
        if result.get('metrics_data'):
            for metric in result['metrics_data']:
                metric_name = metric.get('name', 'Unknown')
                score = metric.get('score')
                if score is not None:
                    if metric_name not in metric_scores:
                        metric_scores[metric_name] = []
                    metric_scores[metric_name].append(score)
    
    print("\n📊 Average Scores by Metric:")
    for metric_name, scores in metric_scores.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"  {metric_name}: {avg_score:.3f} ({len(scores)} cases)")
    
    # Analyze reviewer agreement patterns
    print("\n🔍 Reviewer Agreement Analysis:")
    agreement_analysis = {}
    for result in evaluation_results:
        if result.get('additional_metadata'):
            metadata = result['additional_metadata']
            rev1 = metadata.get('rev1', 'Unknown')
            rev2 = metadata.get('rev2', 'Unknown')
            rev_pair = f"{rev1}-{rev2}"
            
            if rev_pair not in agreement_analysis:
                agreement_analysis[rev_pair] = {'count': 0, 'scores': []}
            
            agreement_analysis[rev_pair]['count'] += 1
            
            # Get average score across all metrics for this case
            if result.get('metrics_data'):
                case_scores = [m.get('score', 0) for m in result['metrics_data'] if m.get('score') is not None]
                if case_scores:
                    agreement_analysis[rev_pair]['scores'].append(sum(case_scores) / len(case_scores))
    
    print("  Reviewer Pair Agreement Scores:")
    for rev_pair, data in agreement_analysis.items():
        if data['scores']:
            avg_score = sum(data['scores']) / len(data['scores'])
            print(f"    {rev_pair}: {avg_score:.3f} ({data['count']} cases)")
    
else:
    print("❌ No results were processed successfully")

# Print summary of all files created
print(f"\n📁 All output files for this reviewer comparison evaluation run:")
print(f"  📊 Main results: {results_file}")
print(f"  📊 Results (JSONL): {results_jsonl_file}")
print(f"  🕐 Timestamp: {timestamp}")
print(f"  ❓ Question type: {question_type}")
print(f"  📂 Results directory: {results_dir}/")
print(f"  🏷️  File prefix: deepeval_reviewer_*_{question_type}_{timestamp}")

print("\n✅ DeepEval reviewer comparison evaluation complete!")
print("\n💡 This evaluation helps understand:")
print("   - How well different reviewers agree on answers")
print("   - Which metrics best capture inter-reviewer differences")
print("   - Patterns in reviewer consensus vs. disagreement")
print("   - Quality of reviewer 2 answers relative to reviewer 1")
