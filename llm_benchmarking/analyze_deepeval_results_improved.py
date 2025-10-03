#!/usr/bin/env python3
"""
Improved script to analyze all JSONL files in deepeval-results directory and generate a comprehensive summary CSV.

This script properly handles different comparison types:
- "results" files: LLM vs Rev1 (standard comparison)
- "reviewer" files: Rev1 vs Rev2 (human reviewer comparison) 
- "llmv2_vs_rev3" files: LLM v2 vs Rev3 (second run comparison)
- "GEval" files: GEval evaluation metrics

The script extracts question types from metadata when available and creates a plotting-friendly dataframe.
"""

import json
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Tuple, Any, Optional


def determine_comparison_type(filename: str) -> str:
    """
    Determine the comparison type from filename.
    
    Args:
        filename: The filename to analyze
        
    Returns:
        The comparison type
    """
    base_name = os.path.splitext(filename)[0]
    
    if 'reviewer' in base_name:
        return 'reviewer_comparison'  # Rev1 vs Rev2
    elif 'llmv2_vs_rev3' in base_name:
        return 'llmv2_vs_rev3'  # LLM v2 vs Rev3
    elif 'GEval' in base_name:
        return 'GEval_evaluation'  # GEval metrics
    elif 'results' in base_name and 'reviewer' not in base_name:
        return 'llm_vs_rev1'  # LLM vs Rev1 (standard)
    else:
        return 'unknown'


def get_comparison_label(filename: str) -> str:
    """
    Get the comparison label from filename.
    
    Args:
        filename: The filename to analyze
        
    Returns:
        The comparison label (llmv1_vs_rev1, rev1_vs_rev2, or llmv2_vs_rev3)
    """
    base_name = os.path.splitext(filename)[0]
    
    if 'reviewer' in base_name:
        return 'rev1_vs_rev2'  # Rev1 vs Rev2
    elif 'llmv2_vs_rev3' in base_name:
        return 'llmv2_vs_rev3'  # LLM v2 vs Rev3
    elif 'results' in base_name and 'reviewer' not in base_name:
        return 'llmv1_vs_rev1'  # LLM v1 vs Rev1 (standard)
    else:
        return 'unknown'


def extract_question_type_from_metadata(record: Dict[str, Any]) -> Optional[str]:
    """
    Extract question type from record metadata.
    
    Args:
        record: The JSON record
        
    Returns:
        Question type if found, None otherwise
    """
    # Check additional_metadata first
    if 'additional_metadata' in record:
        metadata = record['additional_metadata']
        if 'question_id' in metadata:
            return metadata['question_id']
    
    # Check if it's a reviewer comparison with question_id in metadata
    if 'additional_metadata' in record and 'question_id' in record['additional_metadata']:
        return record['additional_metadata']['question_id']
    
    return None


def extract_question_type_from_filename(filename: str) -> str:
    """
    Extract question type from filename as fallback.
    
    Args:
        filename: The filename to extract question type from
        
    Returns:
        The question type extracted from filename
    """
    # Remove file extension
    base_name = os.path.splitext(filename)[0]
    
    # Common question types based on observed filenames
    question_types = [
        'bee_species', 'pesticides', 'additional_stressors', 
        'experimental_methodology', 'future_research', 'significance',
        'all_questions'
    ]
    
    for qtype in question_types:
        if qtype in base_name:
            return qtype
    
    # If no specific question type found, try to extract from filename pattern
    match = re.search(r'results_([^_]+)_', base_name)
    if match:
        return match.group(1)
    
    return 'unknown'


def load_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file and return list of records.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the JSON records
    """
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num} in {filepath}: {e}")
                        continue
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    
    return records


def extract_metrics_from_records(records: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Extract metrics data from records.
    
    Args:
        records: List of records from JSONL file
        
    Returns:
        Dictionary mapping metric names to lists of scores
    """
    metrics_data = {}
    
    for record in records:
        if 'metrics_data' in record and isinstance(record['metrics_data'], list):
            for metric in record['metrics_data']:
                if isinstance(metric, dict) and 'name' in metric and 'score' in metric:
                    metric_name = metric['name']
                    score = metric['score']
                    
                    if isinstance(score, (int, float)):
                        if metric_name not in metrics_data:
                            metrics_data[metric_name] = []
                        metrics_data[metric_name].append(float(score))
    
    return metrics_data


def calculate_statistics(scores: List[float]) -> Tuple[float, int, float, float]:
    """
    Calculate statistics for a list of scores.
    
    Args:
        scores: List of numeric scores
        
    Returns:
        Tuple of (average, count, standard_deviation, standard_error)
    """
    if not scores:
        return 0.0, 0, 0.0, 0.0
    
    scores_array = np.array(scores)
    count = len(scores)
    average = np.mean(scores_array)
    std_dev = np.std(scores_array, ddof=1) if count > 1 else 0.0
    std_error = std_dev / np.sqrt(count) if count > 0 else 0.0
    
    return average, count, std_dev, std_error


def analyze_file_by_question_type(filepath: str, filename: str, comparison_type: str) -> List[Dict[str, Any]]:
    """
    Analyze a single file, potentially breaking down by question type if it contains multiple questions.
    
    Args:
        filepath: Path to the JSONL file
        filename: Name of the file
        comparison_type: Type of comparison
        
    Returns:
        List of result dictionaries
    """
    records = load_jsonl_file(filepath)
    if not records:
        return []
    
    # For reviewer files that contain all questions, we need to group by question type
    if comparison_type == 'reviewer_comparison':
        # Group records by question type
        question_groups = {}
        for record in records:
            question_type = extract_question_type_from_metadata(record)
            if question_type:
                if question_type not in question_groups:
                    question_groups[question_type] = []
                question_groups[question_type].append(record)
            else:
                # Fallback to filename-based question type
                question_type = extract_question_type_from_filename(filename)
                if question_type not in question_groups:
                    question_groups[question_type] = []
                question_groups[question_type].append(record)
        
        # Process each question group separately
        all_results = []
        for question_type, question_records in question_groups.items():
            metrics_data = extract_metrics_from_records(question_records)
            
            for metric_name, scores in metrics_data.items():
                average, count, std_dev, std_error = calculate_statistics(scores)
                
                result_row = {
                    'filename': filename,
                    'comparison_type': comparison_type,
                    'comparison': get_comparison_label(filename),
                    'question_type': question_type,
                    'metric': metric_name,
                    'average': round(average, 6),
                    'count': count,
                    'standard_deviation': round(std_dev, 6),
                    'standard_error': round(std_error, 6)
                }
                all_results.append(result_row)
        
        return all_results
    
    else:
        # For other files, process as single group
        question_type = extract_question_type_from_filename(filename)
        metrics_data = extract_metrics_from_records(records)
        
        all_results = []
        for metric_name, scores in metrics_data.items():
            average, count, std_dev, std_error = calculate_statistics(scores)
            
            result_row = {
                'filename': filename,
                'comparison_type': comparison_type,
                'comparison': get_comparison_label(filename),
                'question_type': question_type,
                'metric': metric_name,
                'average': round(average, 6),
                'count': count,
                'standard_deviation': round(std_dev, 6),
                'standard_error': round(std_error, 6)
            }
            all_results.append(result_row)
        
        return all_results


def analyze_all_jsonl_files(results_dir: str) -> pd.DataFrame:
    """
    Analyze all JSONL files in the results directory.
    
    Args:
        results_dir: Path to the deepeval-results directory
        
    Returns:
        DataFrame containing summary statistics
    """
    # Find all JSONL files
    jsonl_pattern = os.path.join(results_dir, "*.jsonl")
    jsonl_files = glob.glob(jsonl_pattern)
    
    if not jsonl_files:
        print(f"No JSONL files found in {results_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(jsonl_files)} JSONL files to analyze")
    
    all_results = []
    
    for filepath in jsonl_files:
        filename = os.path.basename(filepath)
        comparison_type = determine_comparison_type(filename)
        
        print(f"Processing: {filename} (Type: {comparison_type})")
        
        # Analyze the file
        file_results = analyze_file_by_question_type(filepath, filename, comparison_type)
        all_results.extend(file_results)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    if df.empty:
        return df
    
    # Sort by comparison_type, question_type, and metric for consistent ordering
    df = df.sort_values(['comparison_type', 'question_type', 'metric']).reset_index(drop=True)
    
    return df


def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional summary statistics for plotting.
    
    Args:
        df: The main results DataFrame
        
    Returns:
        DataFrame with additional summary columns
    """
    if df.empty:
        return df
    
    # Add some additional columns for easier plotting
    df['comparison_label'] = df['comparison_type'].map({
        'llm_vs_rev1': 'LLM vs Rev1',
        'reviewer_comparison': 'Rev1 vs Rev2', 
        'llmv2_vs_rev3': 'LLM v2 vs Rev3',
        'GEval_evaluation': 'GEval Evaluation'
    })
    
    # Create a combined identifier for easier grouping
    df['comparison_question'] = df['comparison'] + '_' + df['question_type']
    
    return df


def main():
    """Main function to run the analysis."""
    # Set up paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / "deepeval-results"
    output_file = results_dir / "summary_all_results_improved.csv"
    
    print("DeepEval Results Analysis (Improved)")
    print("=" * 50)
    print(f"Results directory: {results_dir}")
    print(f"Output file: {output_file}")
    print()
    
    # Check if results directory exists
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return
    
    # Analyze all JSONL files
    df = analyze_all_jsonl_files(str(results_dir))
    
    if df.empty:
        print("No data to save")
        return
    
    # Add summary statistics
    df = create_summary_statistics(df)
    
    # Save results
    df.to_csv(output_file, index=False)
    
    print(f"\nAnalysis complete!")
    print(f"Total records processed: {len(df)}")
    print(f"Unique files analyzed: {df['filename'].nunique()}")
    print(f"Unique comparison types: {df['comparison_type'].nunique()}")
    print(f"Unique question types: {df['question_type'].nunique()}")
    print(f"Unique metrics: {df['metric'].nunique()}")
    print(f"Results saved to: {output_file}")
    
    # Display summary by comparison type
    print("\nSummary by Comparison Type:")
    print("-" * 40)
    comparison_summary = df.groupby('comparison_type').agg({
        'filename': 'nunique',
        'question_type': 'nunique',
        'metric': 'nunique',
        'count': 'sum'
    }).rename(columns={
        'filename': 'files',
        'question_type': 'questions',
        'metric': 'metrics',
        'count': 'total_samples'
    })
    print(comparison_summary)
    
    # Display summary by question type
    print("\nSummary by Question Type:")
    print("-" * 30)
    question_summary = df.groupby('question_type').agg({
        'comparison_type': 'nunique',
        'metric': 'nunique',
        'count': 'sum'
    }).rename(columns={
        'comparison_type': 'comparisons',
        'metric': 'metrics',
        'count': 'total_samples'
    })
    print(question_summary)
    
    # Display summary by metric
    print("\nSummary by Metric:")
    print("-" * 20)
    metric_summary = df.groupby('metric').agg({
        'comparison_type': 'nunique',
        'question_type': 'nunique',
        'average': 'mean',
        'count': 'sum'
    }).rename(columns={
        'comparison_type': 'comparisons',
        'question_type': 'questions',
        'average': 'overall_avg',
        'count': 'total_samples'
    }).round(4)
    print(metric_summary)
    
    # Show sample of the data
    print("\nSample of results:")
    print("-" * 20)
    print(df[['comparison_type', 'question_type', 'metric', 'average', 'count']].head(10))


if __name__ == "__main__":
    main()
