#!/usr/bin/env python3
"""
Evaluate LLM v2 answers with DeepEval using Faithfulness, Completeness, and Accuracy metrics.
This script:
1. Loads merged data from final_merged_data/ directory
2. Creates test cases comparing answer_llm2 with answer_rev3
3. Retrieves text chunks from /Users/user/Documents/MetaBeeAI_dataset2/papers
4. Evaluates using Faithfulness, Completeness, and Accuracy metrics
5. Saves results with "llmv2_vs_rev3" identifier for differentiation
"""

import json
import os
import sys
import argparse
from dotenv import load_dotenv
from pathlib import Path
from typing import List

# Load environment variables from .env file
load_dotenv()

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Evaluate LLM v2 answers with DeepEval using specific metrics')
parser.add_argument('--question', '-q', 
                   choices=['bee_species', 'pesticides', 'additional_stressors', 'experimental_methodology', 'significance', 'future_research'],
                   help='Question type to filter by (optional - if not specified, processes all questions)')
parser.add_argument('--limit', '-l', type=int, 
                   help='Maximum number of test cases to process per question (optional)')
parser.add_argument('--batch-size', '-b', type=int, default=25,
                   help='Number of test cases to process per batch (default: 25)')
parser.add_argument('--max-retries', '-r', type=int, default=5,
                   help='Maximum retries per batch (default: 5)')
parser.add_argument('--model', '-m', type=str, default='dual',
                   choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo', 'dual'],
                   help='OpenAI model to use for evaluation (default: dual - GPT-4o-mini for GEval, GPT-4o for Faithfulness/Contextual)')
parser.add_argument('--include-faithfulness', action='store_true',
                   help='Include FaithfulnessMetric in evaluation (slower but more comprehensive)')
parser.add_argument('--include-contextual', action='store_true',
                   help='Include ContextualPrecisionMetric and ContextualRecallMetric in evaluation')
parser.add_argument('--merged-data-dir', type=str, default='llm_benchmarking/final_merged_data',
                   help='Path to final merged data directory (default: llm_benchmarking/final_merged_data)')

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
from deepeval.metrics import FaithfulnessMetric, GEval, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig, DisplayConfig
from deepeval.models import GPTModel

def load_merged_data_files(merged_data_dir):
    """Load all merged data files from the specified directory"""
    merged_data = {}
    
    if not os.path.exists(merged_data_dir):
        raise ValueError(f"Merged data directory not found: {merged_data_dir}")
    
    # Get all merged JSON files
    merged_files = [f for f in os.listdir(merged_data_dir) if f.endswith('_merged.json')]
    
    if not merged_files:
        raise ValueError(f"No merged JSON files found in {merged_data_dir}")
    
    print(f"📁 Loading merged data files from: {merged_data_dir}")
    
    for merged_file in merged_files:
        question_name = merged_file.replace('_merged.json', '')
        merged_file_path = os.path.join(merged_data_dir, merged_file)
        
        try:
            with open(merged_file_path, 'r') as f:
                data = json.load(f)
                merged_data[question_name] = data
                print(f"  ✅ Loaded {question_name}: {len(data)} papers")
        except Exception as e:
            print(f"  ⚠️  Error loading {merged_file}: {e}")
    
    return merged_data

def get_chunk_ids_for_question(paper_id: str, question_name: str, dataset2_dir: str) -> List[str]:
    """Get chunk IDs for a specific question from answers.json in dataset2."""
    answers_file = os.path.join(dataset2_dir, 'papers', paper_id, "answers.json")
    
    try:
        with open(answers_file, 'r', encoding='utf-8') as f:
            answers_data = json.load(f)
        
        if 'QUESTIONS' in answers_data:
            questions = answers_data['QUESTIONS']
            
            # Direct lookup first
            if question_name in questions:
                question_data = questions[question_name]
                if isinstance(question_data, dict) and "chunk_ids" in question_data:
                    return question_data["chunk_ids"]
            
            # Also check grouped questions (like bee_and_pesticides.bee_species)
            for group_key, group_data in questions.items():
                if isinstance(group_data, dict):
                    for sub_key, sub_data in group_data.items():
                        if sub_key == question_name:
                            if isinstance(sub_data, dict) and "chunk_ids" in sub_data:
                                return sub_data["chunk_ids"]
                                
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"  ⚠️  Could not load chunk IDs for paper {paper_id}, question {question_name}: {e}")
    
    return []

def get_text_chunks(paper_id: str, chunk_ids: List[str], dataset2_dir: str) -> List[str]:
    """Get text chunks from merged_v2.json based on chunk IDs."""
    merged_file = os.path.join(dataset2_dir, 'papers', paper_id, "pages", "merged_v2.json")
    
    # Fallback to merged.json if merged_v2.json doesn't exist
    if not os.path.exists(merged_file):
        merged_file = os.path.join(dataset2_dir, 'papers', paper_id, "pages", "merged.json")
    
    text_chunks = []
    
    try:
        with open(merged_file, 'r', encoding='utf-8') as f:
            merged_data = json.load(f)
            
        # Navigate the structure: merged_data -> data -> chunks -> list of chunk objects
        if isinstance(merged_data, dict) and "data" in merged_data:
            data = merged_data["data"]
            if isinstance(data, dict) and "chunks" in data:
                chunks = data["chunks"]
                
                if isinstance(chunks, list):
                    # Chunks is a list of chunk objects
                    for chunk in chunks:
                        if isinstance(chunk, dict):
                            chunk_id = chunk.get("chunk_id", "")
                            if chunk_id in chunk_ids:
                                text = chunk.get("text", "")
                                if text and text not in text_chunks:
                                    text_chunks.append(text)
                elif isinstance(chunks, dict):
                    # Chunks is a dict with chunk_id as keys
                    for chunk_id, chunk_data in chunks.items():
                        if chunk_id in chunk_ids:
                            if isinstance(chunk_data, dict) and "text" in chunk_data:
                                text = chunk_data["text"]
                                if text and text not in text_chunks:
                                    text_chunks.append(text)
                            
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  ⚠️  Could not load merged data for paper {paper_id}: {e}")
    
    return text_chunks

def create_test_cases_from_merged_data(merged_data, question_filter=None, limit_per_question=None, dataset2_dir=None):
    """Create test cases from merged data comparing answer_llm2 with answer_rev3"""
    test_cases = []
    skipped_count = 0
    
    # Filter questions if specified
    questions_to_process = [question_filter] if question_filter else list(merged_data.keys())
    
    print(f"📊 Creating test cases for questions: {questions_to_process}")
    
    for question_name in questions_to_process:
        if question_name not in merged_data:
            print(f"⚠️  Question '{question_name}' not found in merged data")
            continue
        
        question_data = merged_data[question_name]
        question_test_cases = []
        
        print(f"\n🔄 Processing {question_name}: {len(question_data)} papers")
        
        for paper_id, paper_data in question_data.items():
            # Check if answer_llm2 exists
            if 'answer_llm2' not in paper_data:
                print(f"  ⚠️  Skipping paper {paper_id}: No answer_llm2 found")
                skipped_count += 1
                continue
            
            answer_llm2 = paper_data['answer_llm2']
            if not answer_llm2 or answer_llm2.strip() == "":
                print(f"  ⚠️  Skipping paper {paper_id}: Empty answer_llm2")
                skipped_count += 1
                continue
            
            # Get answer_rev3 as the reviewer answer to compare against
            reviewer_answer = ""
            reviewer_id = "rev3"
            reviewer_rating = ""
            
            if 'answer_rev3' in paper_data and paper_data['answer_rev3']:
                reviewer_answer = paper_data['answer_rev3']
                reviewer_rating = paper_data.get('rev3_rating', '')
            
            if not reviewer_answer or reviewer_answer.strip() == "":
                print(f"  ⚠️  Skipping paper {paper_id}: No answer_rev3 found")
                skipped_count += 1
                continue
            
            # Create input question based on question type
            question_prompts = {
                'bee_species': 'What species of bee(s) were tested?',
                'pesticides': 'What pesticides were used, what doses, what was the exposure method, and what was the duration of exposure?',
                'additional_stressors': 'Were any additional stressors or stressor combinations used? If yes, what was the dose, exposure method, and duration of exposure?',
                'experimental_methodology': 'What was the experimental methodology used in this study?',
                'significance': 'What is the significance of this study?',
                'future_research': 'What future research directions are suggested?'
            }
            
            input_question = question_prompts.get(question_name, f"Answer the question about {question_name}")
            
            # Get context for this question if dataset2_dir is provided
            context = None
            if dataset2_dir:
                try:
                    chunk_ids = get_chunk_ids_for_question(paper_id, question_name, dataset2_dir)
                    if chunk_ids:
                        text_chunks = get_text_chunks(paper_id, chunk_ids, dataset2_dir)
                        if text_chunks:
                            context = text_chunks
                            print(f"  📄 Retrieved {len(text_chunks)} context chunks for paper {paper_id}")
                        else:
                            print(f"  ⚠️  No text chunks found for paper {paper_id}, question {question_name}")
                    else:
                        print(f"  ⚠️  No chunk IDs found for paper {paper_id}, question {question_name}")
                except Exception as e:
                    print(f"  ⚠️  Error retrieving context for paper {paper_id}: {e}")
            
            # Create test case
            try:
                test_case_kwargs = {
                    "input": input_question,
                    "actual_output": answer_llm2,  # LLM v2 answer
                    "expected_output": reviewer_answer,  # Reviewer answer (gold standard)
                    "name": f"paper_{paper_id}_{question_name}",
                    "additional_metadata": {
                        "paper_id": paper_id,
                        "question_id": question_name,
                        "reviewer": reviewer_id,
                        "reviewer_rating": reviewer_rating,
                        "llm_version": "v2",
                        "comparison_type": "llm2_vs_rev3"
                    }
                }
                
                # Only add context if available and FaithfulnessMetric is requested
                if context and len(context) > 0 and args.include_faithfulness:
                    test_case_kwargs["context"] = context
                    test_case_kwargs["retrieval_context"] = context  # FaithfulnessMetric requires this
                    print(f"  📄 Added context for paper {paper_id} ({len(context)} chunks)")
                elif context and len(context) > 0:
                    print(f"  📄 Context available for paper {paper_id} ({len(context)} chunks) - use --include-faithfulness to enable")
                
                test_case = LLMTestCase(**test_case_kwargs)
                
                # Set proper identifiers
                test_case._identifier = f"paper_{paper_id}_{question_name}_llmv2"
                test_case._dataset_id = f"metabeeai_llmv2_{question_name}"
                
                question_test_cases.append(test_case)
                
            except Exception as e:
                print(f"  ⚠️  Error creating test case for paper {paper_id}: {e}")
                skipped_count += 1
                continue
        
        # Apply limit if specified
        if limit_per_question and len(question_test_cases) > limit_per_question:
            question_test_cases = question_test_cases[:limit_per_question]
            print(f"  📊 Limited to first {limit_per_question} test cases for {question_name}")
        
        test_cases.extend(question_test_cases)
        print(f"  ✅ Created {len(question_test_cases)} test cases for {question_name}")
    
    if skipped_count > 0:
        print(f"\n⚠️  Skipped {skipped_count} test cases due to missing data")
    
    return test_cases

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

def process_test_cases_in_batches(test_cases, metrics, batch_size=25, max_retries=5, results_file=None, results_jsonl_file=None):
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
                        max_concurrent=3   # Lower concurrency for better stability
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
                    if results_file and results_jsonl_file:
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

def main():
    # Convert relative paths to absolute
    if not os.path.isabs(args.merged_data_dir):
        script_dir = Path(__file__).parent.parent  # Go up to pipeline root
        merged_data_dir = script_dir / args.merged_data_dir
    else:
        merged_data_dir = args.merged_data_dir
    
    # Set dataset2 directory for context retrieval
    dataset2_dir = "/Users/user/Documents/MetaBeeAI_dataset2"
    if not os.path.exists(dataset2_dir):
        print(f"⚠️  Warning: Dataset2 directory not found: {dataset2_dir}")
        print("   FaithfulnessMetric will not work properly without context.")
        dataset2_dir = None
    
    # Load merged data
    print("🔄 Loading merged data files...")
    merged_data = load_merged_data_files(merged_data_dir)
    
    # Create test cases
    print("\n🔄 Creating test cases from merged data...")
    test_cases = create_test_cases_from_merged_data(
        merged_data, 
        question_filter=args.question,
        limit_per_question=args.limit,
        dataset2_dir=dataset2_dir
    )
    
    if not test_cases:
        print("❌ No test cases created. Exiting.")
        return
    
    print(f"\n✅ Created {len(test_cases)} test cases total")
    
    # Create dataset
    dataset = EvaluationDataset()
    for test_case in test_cases:
        dataset.add_test_case(test_case)
    
    # Configure models based on dual-model approach or single model override
    if args.model == 'dual':
        geval_model = GPTModel(model='gpt-4o-mini')  # Cost-effective for GEval metrics
        contextual_model = GPTModel(model='gpt-4o')  # Higher quality for Faithfulness/Contextual metrics
        print("🔄 Using dual-model approach:")
        print("  📊 GEval metrics (Completeness, Accuracy): GPT-4o-mini")
        print("  🎯 Contextual metrics (Faithfulness, Contextual Precision/Recall): GPT-4o")
    else:
        # Single model for all metrics
        geval_model = GPTModel(model=args.model)
        contextual_model = GPTModel(model=args.model)
        print(f"🔄 Using single model for all metrics: {args.model}")
    
    # Define base metrics (always included)
    metrics = [
        GEval(
            name="Completeness",
            criteria="Completeness - assess if the actual output covers all the key points mentioned in the expected output.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model=geval_model,
            strict_mode=False
        ),
        GEval(
            name="Accuracy",
            criteria="Accuracy - evaluate if the actual output contains accurate information that aligns with the expected output.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model=geval_model,
            strict_mode=False
        )
    ]
    
    # Add FaithfulnessMetric only if requested (it's much slower)
    if args.include_faithfulness:
        metrics.insert(0, FaithfulnessMetric(model=contextual_model))
        print("✅ FaithfulnessMetric included (slower but more comprehensive)")
    else:
        print("ℹ️  FaithfulnessMetric skipped (use --include-faithfulness to enable)")
    
    # Add Contextual Precision and Recall metrics if requested
    if args.include_contextual:
        metrics.append(ContextualPrecisionMetric(model=contextual_model))
        metrics.append(ContextualRecallMetric(model=contextual_model))
        print("✅ ContextualPrecisionMetric and ContextualRecallMetric included")
    else:
        print("ℹ️  Contextual metrics skipped (use --include-contextual to enable)")
    
    # Show cost comparison
    cost_info = {
        'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006, 'description': 'Most cost-effective'},
        'gpt-4o': {'input': 0.0025, 'output': 0.01, 'description': 'Balanced performance/cost'},
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03, 'description': 'Higher cost, better performance'},
        'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015, 'description': 'Good cost, lower performance'}
    }
    
    if args.model == 'dual':
        geval_cost = cost_info['gpt-4o-mini']
        contextual_cost = cost_info['gpt-4o']
        print(f"💰 GEval metrics cost per 1K tokens: Input ${geval_cost['input']:.4f}, Output ${geval_cost['output']:.4f}")
        print(f"💰 Contextual metrics cost per 1K tokens: Input ${contextual_cost['input']:.4f}, Output ${contextual_cost['output']:.4f}")
        print(f"💰 Dual-model approach balances cost-effectiveness with quality")
    else:
        selected_cost = cost_info[args.model]
        print(f"💰 Cost per 1K tokens: Input ${selected_cost['input']:.4f}, Output ${selected_cost['output']:.4f}")
        print(f"💰 Recommendation: {selected_cost['description']}")
    
    print(f"\n📊 Evaluating with {len(metrics)} metrics:")
    for metric in metrics:
        if hasattr(metric, 'name'):
            print(f"  - {metric.name}")
        else:
            print(f"  - {metric.__class__.__name__}")
    
    # Initialize results tracking with timestamped filenames
    import datetime
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    question_type = args.question if args.question else "all_questions"
    
    # Create results directory if it doesn't exist
    results_dir = "llm_benchmarking/deepeval-results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate unique filenames in the results directory with "llmv2_vs_rev3" identifier
    results_file = f"{results_dir}/deepeval_llmv2_vs_rev3_results_{question_type}_{timestamp}.json"
    results_jsonl_file = f"{results_dir}/deepeval_llmv2_vs_rev3_results_{question_type}_{timestamp}.jsonl"
    
    print(f"\n📁 Output files will be saved in: {results_dir}/")
    print(f"📁 File prefix: deepeval_llmv2_vs_rev3_*_{question_type}_{timestamp}")
    
    # Run evaluation with batch processing and incremental saving
    print(f"\n🚀 Starting LLM v2 evaluation of {len(dataset.test_cases)} test cases...")
    print("⏳ Processing in batches with incremental saving to prevent data loss...")
    
    # Process test cases in batches with incremental saving
    evaluation_results = process_test_cases_in_batches(
        dataset.test_cases, 
        metrics,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        results_file=results_file,
        results_jsonl_file=results_jsonl_file
    )
    
    if evaluation_results:
        print(f"\n🎉 LLM v2 evaluation completed successfully!")
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
    
    # Print summary of all files created
    print(f"\n📁 All output files for this LLM v2 vs Rev3 evaluation run:")
    print(f"  📊 Main results: {results_file}")
    print(f"  📊 Results (JSONL): {results_jsonl_file}")
    print(f"  🕐 Timestamp: {timestamp}")
    print(f"  ❓ Question type: {question_type}")
    print(f"  📂 Results directory: {results_dir}/")
    
    print("\n✅ DeepEval LLM v2 vs Rev3 evaluation complete!")

if __name__ == "__main__":
    main()
