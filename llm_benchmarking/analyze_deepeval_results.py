#!/usr/bin/env python3
"""
Script to analyze all JSONL files in deepeval-results directory and generate a comprehensive summary CSV.

This script extracts metrics data from all JSONL files, calculates statistics (average, count, 
standard deviation, standard error) for each metric, and saves the results as summary_all_results.csv.
"""

import json
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Tuple, Any


def extract_question_type_from_filename(filename: str) -> str:
    """
    Extract question type from filename.
    
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
    # Look for patterns like "results_[question_type]_"
    match = re.search(r'results_([^_]+)_', base_name)
    if match:
        return match.group(1)
    
    return 'unknown'


def extract_experiment_type_from_filename(filename: str) -> str:
    """
    Extract experiment type from filename.
    
    Args:
        filename: The filename to extract experiment type from
        
    Returns:
        The experiment type
    """
    base_name = os.path.splitext(filename)[0]
    
    if 'GEval' in base_name:
        return 'GEval'
    elif 'llmv2_vs_rev3' in base_name:
        return 'llmv2_vs_rev3'
    elif 'reviewer' in base_name:
        return 'reviewer'
    else:
        return 'standard'


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
        print(f"Processing: {filename}")
        
        # Extract metadata from filename
        question_type = extract_question_type_from_filename(filename)
        experiment_type = extract_experiment_type_from_filename(filename)
        
        # Load and process the file
        records = load_jsonl_file(filepath)
        if not records:
            print(f"Warning: No records found in {filename}")
            continue
        
        # Extract metrics
        metrics_data = extract_metrics_from_records(records)
        
        if not metrics_data:
            print(f"Warning: No metrics data found in {filename}")
            continue
        
        # Calculate statistics for each metric
        for metric_name, scores in metrics_data.items():
            average, count, std_dev, std_error = calculate_statistics(scores)
            
            result_row = {
                'filename': filename,
                'question_type': question_type,
                'experiment_type': experiment_type,
                'metric': metric_name,
                'average': round(average, 6),
                'count': count,
                'standard_deviation': round(std_dev, 6),
                'standard_error': round(std_error, 6)
            }
            
            all_results.append(result_row)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by filename, question_type, and metric for consistent ordering
    df = df.sort_values(['filename', 'question_type', 'metric']).reset_index(drop=True)
    
    return df


def main():
    """Main function to run the analysis."""
    # Set up paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / "deepeval-results"
    output_file = results_dir / "summary_all_results.csv"
    
    print("DeepEval Results Analysis")
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
    
    # Save results
    df.to_csv(output_file, index=False)
    
    print(f"\nAnalysis complete!")
    print(f"Total records processed: {len(df)}")
    print(f"Unique files analyzed: {df['filename'].nunique()}")
    print(f"Unique question types: {df['question_type'].nunique()}")
    print(f"Unique metrics: {df['metric'].nunique()}")
    print(f"Results saved to: {output_file}")
    
    # Display summary
    print("\nSummary by Question Type:")
    print("-" * 30)
    summary = df.groupby('question_type').agg({
        'filename': 'nunique',
        'metric': 'nunique',
        'count': 'sum'
    }).rename(columns={
        'filename': 'files',
        'metric': 'metrics',
        'count': 'total_samples'
    })
    print(summary)
    
    print("\nSummary by Metric:")
    print("-" * 20)
    metric_summary = df.groupby('metric').agg({
        'filename': 'nunique',
        'average': 'mean',
        'count': 'sum'
    }).rename(columns={
        'filename': 'files',
        'average': 'overall_avg',
        'count': 'total_samples'
    }).round(4)
    print(metric_summary)


if __name__ == "__main__":
    main()
