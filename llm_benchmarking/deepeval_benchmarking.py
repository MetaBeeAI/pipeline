import json
import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Evaluate test dataset with DeepEval locally')
parser.add_argument('--question', '-q', 
                   choices=['bee_species', 'pesticides', 'additional_stressors', 'experimental_methodology', 'significance', 'future_research', 'limitations'],
                   help='Question type to filter by (optional - if not specified, processes all questions)')
parser.add_argument('--limit', '-l', type=int, 
                   help='Maximum number of test cases to process (optional)')
parser.add_argument('--batch-size', '-b', type=int, default=50,
                   help='Number of test cases to process per batch (default: 50). Use smaller batches (5-10) for papers with very long context to avoid token limit errors.')
parser.add_argument('--max-retries', '-r', type=int, default=5,
                   help='Maximum retries per batch (default: 5)')
parser.add_argument('--model', '-m', type=str, default='gpt-4o-mini',
                   choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo'],
                   help='OpenAI model to use for evaluation (default: gpt-4o-mini for cost reduction)')
parser.add_argument('--max-context-length', type=int, default=60000,
                   help='Maximum context length in characters to process (default: 60000, use smaller values for problematic papers)')

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
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig, DisplayConfig
from deepeval.models import GPTModel

# Load your test dataset
with open("llm_benchmarking/test-datasets/test_dataset.json", "r") as f:
    data = json.load(f)

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
    required_fields = ["input", "actual_outputs", "expected_output", "context"]
    missing_fields = [field for field in required_fields if not entry.get(field)]
    
    if missing_fields:
        print(f"⚠️  Skipping test case {i+1}: Missing fields {missing_fields}")
        skipped_count += 1
        continue
    
    # Get retrieval_context, use context as fallback if missing
    retrieval_context = entry.get("retrieval_context")
    if not retrieval_context:
        retrieval_context = entry["context"]  # Use context as fallback
    
    # Check context length to avoid token limit issues
    context_length = len(str(entry["context"]))
    
    if context_length > args.max_context_length:
        print(f"⚠️  Skipping test case {i+1}: Context too long ({context_length:,} chars, max: {args.max_context_length:,})")
        skipped_count += 1
        continue
    
    # Truncate context if it's still very long to prevent token limit issues during evaluation
    if context_length > 80000:  # Higher threshold for gpt-4o
        print(f"⚠️  Truncating context for test case {i+1}: {context_length:,} chars -> 80,000 chars")
        entry["context"] = str(entry["context"])[:80000] + "... [truncated]"
        if retrieval_context:
            retrieval_context = str(retrieval_context)[:80000] + "... [truncated]"
    
    try:
        # Create LLMTestCase object with proper identifiers
        test_case = LLMTestCase(
            input=entry["input"],
            actual_output=entry["actual_outputs"],  # LLM generated answer
            expected_output=entry["expected_output"],  # Human reviewer answer (gold standard)
            context=entry["context"],  # Full paper context
            retrieval_context=retrieval_context,  # Retrieval context for metrics that need it
            name=f"paper_{entry['id']}_case_{i}",  # Unique name for each test case
            additional_metadata={
                "paper_id": entry["id"],
                "question_id": entry["metadata"]["question_id"],
                "reviewer": entry["metadata"]["reviewer"],
                "rating": entry["metadata"]["rating"]
            }
        )
        
        # Set proper identifiers to avoid ID errors
        test_case._identifier = f"paper_{entry['id']}_case_{i}"
        test_case._dataset_id = f"metabeeai_{args.question if args.question else 'all'}"
        
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

# Warn about long contexts and recommend settings
long_context_count = sum(1 for entry in filtered_data if len(str(entry.get("context", ""))) > 50000)
if long_context_count > 0:
    print(f"⚠️  WARNING: {long_context_count} test cases have very long context (>50K chars)")
    print(f"💡 RECOMMENDED: Use --batch-size 15-25 and --max-context-length 60000 for gpt-4o")
    print(f"💡 Example: python llm_benchmarking/deepeval_benchmarking.py --question {args.question} --model gpt-4o --batch-size 25 --max-context-length 60000")

# Configure the specified model (default: GPT-4o-mini for cost reduction)
evaluation_model = GPTModel(model=args.model)

# Define metrics to evaluate (all using the specified model)
metrics = [
    FaithfulnessMetric(model=evaluation_model),              # ⭐⭐⭐⭐⭐ Measures how faithful is the answer to the expected output
    ContextualPrecisionMetric(model=evaluation_model),       # ⭐⭐⭐⭐ Measures precision in the context of expected answer
    ContextualRecallMetric(model=evaluation_model)           # ⭐⭐⭐⭐ Measures recall in the context of expected answer
]

print(f"💰 Using {args.model} for evaluation (cost-effective option)")

# Show cost comparison
cost_info = {
    'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006, 'description': 'Most cost-effective'},
    'gpt-4o': {'input': 0.0025, 'output': 0.01, 'description': 'Balanced performance/cost'},
    'gpt-4-turbo': {'input': 0.01, 'output': 0.03, 'description': 'Higher cost, better performance'},
    'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015, 'description': 'Good cost, lower performance'}
}

selected_cost = cost_info[args.model]
print(f"💰 Cost per 1K tokens: Input ${selected_cost['input']:.4f}, Output ${selected_cost['output']:.4f}")
print(f"💰 Recommendation: {selected_cost['description']}")

print(f"Evaluating with {len(metrics)} metrics:")
for metric in metrics:
    print(f"  - {metric.__class__.__name__}")

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
results_file = f"{results_dir}/deepeval_results_{question_type}_{timestamp}.json"
results_jsonl_file = f"{results_dir}/deepeval_results_{question_type}_{timestamp}.jsonl"

print(f"📁 Output files will be saved in: {results_dir}/")
print(f"📁 File prefix: deepeval_*_{question_type}_{timestamp}")

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
    
    # Note: Context truncation removed - keeping full context for better evaluation quality
    
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
                
                # Process this batch with full context (no truncation)
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
                                "context": getattr(r, 'context', None),
                                "retrieval_context": getattr(r, 'retrieval_context', None),
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
                
                # Check if it's an API token limit issue
                if "length limit was reached" in error_msg or "token limit" in error_msg.lower():
                    print(f"⚠️  API token limit reached for batch {batch_start//batch_size + 1} (attempt {retry_count}/{max_retries})")
                    print(f"💡 This batch has very long context - consider reducing batch size")
                else:
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
print(f"\n🚀 Starting evaluation of {len(dataset.test_cases)} test cases...")
print("⏳ Processing in batches with incremental saving to prevent data loss...")

# Process test cases in batches with incremental saving
batch_size = args.batch_size  # Process test cases per batch
max_retries = args.max_retries  # Maximum retries per batch
evaluation_results = process_test_cases_in_batches(dataset.test_cases, batch_size, max_retries)

if evaluation_results:
    print(f"\n🎉 Evaluation completed successfully!")
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
    
else:
    print("❌ No results were processed successfully")

# Note: Fallback results removed as they were redundant with the main results

# Note: Test cases summary removed as it was redundant with the main results

# Print summary of all files created
print(f"\n📁 All output files for this evaluation run:")
print(f"  📊 Main results: {results_file}")
print(f"  📊 Results (JSONL): {results_jsonl_file}")
print(f"  🕐 Timestamp: {timestamp}")
print(f"  ❓ Question type: {question_type}")
print(f"  📂 Results directory: {results_dir}/")

print("\n✅ DeepEval evaluation complete!")






