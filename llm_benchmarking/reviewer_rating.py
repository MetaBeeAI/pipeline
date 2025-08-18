#!/usr/bin/env python3
"""
Reviewer Rating Analysis Script

This script analyzes reviewer ratings from the final_merged_data directory and generates:
1. Average reviewer rating scores for each question with standard errors
2. Average reviewer agreement (difference between rev1 and rev2 ratings)
3. Average rating per individual reviewer

Author: MetaBeeAI Pipeline
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
except:
    # Fallback if seaborn is not available
    plt.style.use('default')

def load_merged_data(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all merged JSON files from the data directory."""
    data = {}
    data_path = Path(data_dir)
    
    for json_file in data_path.glob("*.json"):
        question_name = json_file.stem.replace("_merged", "")
        with open(json_file, 'r') as f:
            data[question_name] = json.load(f)
    
    return data

def filter_valid_ratings(rating: int) -> bool:
    """Filter out ratings that are 0 (invalid)."""
    return rating > 0

def calculate_question_stats(data: Dict[str, Any]) -> Tuple[float, float, int]:
    """Calculate average rating, standard error, and count for a question."""
    ratings = []
    
    for paper_id, paper_data in data.items():
        # Check if both reviewers have valid ratings
        if 'rev1_rating' in paper_data and 'rev2_rating' in paper_data:
            rev1_rating = paper_data['rev1_rating']
            rev2_rating = paper_data['rev2_rating']
            
            if filter_valid_ratings(rev1_rating):
                ratings.append(rev1_rating)
            if filter_valid_ratings(rev2_rating):
                ratings.append(rev2_rating)
    
    if not ratings:
        return 0.0, 0.0, 0
    
    avg_rating = np.mean(ratings)
    std_error = np.std(ratings) / np.sqrt(len(ratings))
    count = len(ratings)
    
    return avg_rating, std_error, count

def calculate_reviewer_agreement(data: Dict[str, Any]) -> Tuple[float, float, int]:
    """Calculate average agreement (difference) between rev1 and rev2 ratings."""
    agreements = []
    
    for paper_id, paper_data in data.items():
        if 'rev1_rating' in paper_data and 'rev2_rating' in paper_data:
            rev1_rating = paper_data['rev1_rating']
            rev2_rating = paper_data['rev2_rating']
            
            # Only include papers where both reviewers have valid ratings
            if filter_valid_ratings(rev1_rating) and filter_valid_ratings(rev2_rating):
                agreement = abs(rev1_rating - rev2_rating)
                agreements.append(agreement)
    
    if not agreements:
        return 0.0, 0.0, 0
    
    avg_agreement = np.mean(agreements)
    std_error = np.std(agreements) / np.sqrt(len(agreements))
    count = len(agreements)
    
    return avg_agreement, std_error, count

def calculate_reviewer_individual_stats(data: Dict[str, Any]) -> Dict[str, List[int]]:
    """Calculate individual reviewer statistics."""
    reviewer_ratings = {}
    
    for paper_id, paper_data in data.items():
        if 'rev1' in paper_data and 'rev1_rating' in paper_data:
            reviewer = paper_data['rev1']
            rating = paper_data['rev1_rating']
            if filter_valid_ratings(rating):
                if reviewer not in reviewer_ratings:
                    reviewer_ratings[reviewer] = []
                reviewer_ratings[reviewer].append(rating)
        
        if 'rev2' in paper_data and 'rev2_rating' in paper_data:
            reviewer = paper_data['rev2']
            rating = paper_data['rev2_rating']
            if filter_valid_ratings(rating):
                if reviewer not in reviewer_ratings:
                    reviewer_ratings[reviewer] = []
                reviewer_ratings[reviewer].append(rating)
    
    return reviewer_ratings

def plot_question_ratings(question_stats: Dict[str, Tuple[float, float, int]], output_path: str):
    """Create bar plot of average ratings per question with standard error bars."""
    questions = list(question_stats.keys())
    avg_ratings = [question_stats[q][0] for q in questions]
    std_errors = [question_stats[q][1] for q in questions]
    counts = [question_stats[q][2] for q in questions]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bars
    bars = ax.bar(questions, avg_ratings, yerr=std_errors, capsize=5, 
                  alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Question Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Rating (1-10)', fontsize=14, fontweight='bold')
    ax.set_title('Average Reviewer Ratings by Question Type\n(with Standard Errors)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add value labels on bars
    for i, (bar, avg, count) in enumerate(zip(bars, avg_ratings, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_errors[i] + 0.1,
                f'{avg:.2f}\n(n={count})', ha='center', va='bottom', fontsize=10)
    
    # Set y-axis limits
    ax.set_ylim(0, max(avg_ratings) + max(std_errors) + 1)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Question ratings plot saved to: {output_path}")

def plot_reviewer_agreement(agreement_stats: Dict[str, Tuple[float, float, int]], output_path: str):
    """Create bar plot of reviewer agreement (lower = better agreement)."""
    questions = list(agreement_stats.keys())
    avg_agreements = [agreement_stats[q][0] for q in questions]
    std_errors = [agreement_stats[q][1] for q in questions]
    counts = [agreement_stats[q][2] for q in questions]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bars (note: lower values indicate better agreement)
    bars = ax.bar(questions, avg_agreements, yerr=std_errors, capsize=5,
                  alpha=0.7, color='lightcoral', edgecolor='darkred', linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Question Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Rating Difference (Lower = Better Agreement)', fontsize=14, fontweight='bold')
    ax.set_title('Reviewer Agreement by Question Type\n(Lower Values = Better Agreement)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add value labels on bars
    for i, (bar, avg, count) in enumerate(zip(bars, avg_agreements, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_errors[i] + 0.05,
                f'{avg:.2f}\n(n={count})', ha='center', va='bottom', fontsize=10)
    
    # Set y-axis limits
    ax.set_ylim(0, max(avg_agreements) + max(std_errors) + 0.5)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Reviewer agreement plot saved to: {output_path}")

def plot_individual_reviewer_ratings(reviewer_stats: Dict[str, List[int]], output_path: str):
    """Create bar plot of average ratings per individual reviewer."""
    reviewers = list(reviewer_stats.keys())
    avg_ratings = [np.mean(reviewer_stats[r]) for r in reviewers]
    std_errors = [np.std(reviewer_stats[r]) / np.sqrt(len(reviewer_stats[r])) for r in reviewers]
    counts = [len(reviewer_stats[r]) for r in reviewers]
    
    # Sort by average rating for better visualization
    sorted_indices = np.argsort(avg_ratings)[::-1]  # Sort descending
    reviewers = [reviewers[i] for i in sorted_indices]
    avg_ratings = [avg_ratings[i] for i in sorted_indices]
    std_errors = [std_errors[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bars
    bars = ax.bar(reviewers, avg_ratings, yerr=std_errors, capsize=5,
                  alpha=0.7, color='lightgreen', edgecolor='darkgreen', linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Reviewer Initials', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Rating (1-10)', fontsize=14, fontweight='bold')
    ax.set_title('Average Rating per Individual Reviewer\n(with Standard Errors)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add value labels on bars
    for i, (bar, avg, count) in enumerate(zip(bars, avg_ratings, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_errors[i] + 0.1,
                f'{avg:.2f}\n(n={count})', ha='center', va='bottom', fontsize=10)
    
    # Set y-axis limits
    ax.set_ylim(0, max(avg_ratings) + max(std_errors) + 1)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Individual reviewer ratings plot saved to: {output_path}")

def main():
    """Main function to run the analysis."""
    print("Starting Reviewer Rating Analysis...")
    
    # Define paths
    data_dir = "llm_benchmarking/final_merged_data"
    analyses_dir = "llm_benchmarking/analyses"
    
    # Create analyses directory if it doesn't exist
    Path(analyses_dir).mkdir(exist_ok=True)
    
    # Load data
    print("Loading merged data...")
    data = load_merged_data(data_dir)
    
    if not data:
        print("No data found! Please check the data directory.")
        return
    
    print(f"Loaded data for {len(data)} question types: {list(data.keys())}")
    
    # Calculate statistics for each question
    print("\nCalculating question statistics...")
    question_stats = {}
    agreement_stats = {}
    
    for question_name, question_data in data.items():
        print(f"Processing {question_name}...")
        
        # Calculate average ratings
        avg_rating, std_error, count = calculate_question_stats(question_data)
        question_stats[question_name] = (avg_rating, std_error, count)
        
        # Calculate reviewer agreement
        avg_agreement, agreement_std_error, agreement_count = calculate_reviewer_agreement(question_data)
        agreement_stats[question_name] = (avg_agreement, agreement_std_error, agreement_count)
        
        print(f"  - {question_name}: Avg Rating = {avg_rating:.2f} ± {std_error:.2f} (n={count})")
        print(f"  - {question_name}: Avg Agreement = {avg_agreement:.2f} ± {agreement_std_error:.2f} (n={agreement_count})")
    
    # Calculate individual reviewer statistics
    print("\nCalculating individual reviewer statistics...")
    all_reviewer_ratings = {}
    
    for question_name, question_data in data.items():
        reviewer_ratings = calculate_reviewer_individual_stats(question_data)
        
        for reviewer, ratings in reviewer_ratings.items():
            if reviewer not in all_reviewer_ratings:
                all_reviewer_ratings[reviewer] = []
            all_reviewer_ratings[reviewer].extend(ratings)
    
    # Print individual reviewer stats
    for reviewer, ratings in all_reviewer_ratings.items():
        avg_rating = np.mean(ratings)
        std_error = np.std(ratings) / np.sqrt(len(ratings))
        print(f"  - {reviewer}: Avg Rating = {avg_rating:.2f} ± {std_error:.2f} (n={len(ratings)})")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Plot 1: Average ratings per question
    plot_question_ratings(question_stats, f"{analyses_dir}/avg-rev-ratings.png")
    
    # Plot 2: Reviewer agreement
    plot_reviewer_agreement(agreement_stats, f"{analyses_dir}/avg-rev-agreement.png")
    
    # Plot 3: Individual reviewer ratings
    plot_individual_reviewer_ratings(all_reviewer_ratings, f"{analyses_dir}/avg-score-per-reviewer.png")
    
    print("\nAnalysis complete! All plots have been saved to the analyses directory.")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print("\nQuestion-wise Average Ratings (excluding ratings = 0):")
    for question, (avg, std_err, count) in question_stats.items():
        print(f"  {question:25s}: {avg:6.2f} ± {std_err:5.2f} (n={count:3d})")
    
    print("\nQuestion-wise Reviewer Agreement (lower = better):")
    for question, (avg, std_err, count) in agreement_stats.items():
        print(f"  {question:25s}: {avg:6.2f} ± {std_err:5.2f} (n={count:3d})")
    
    print("\nIndividual Reviewer Average Ratings:")
    for reviewer, ratings in sorted(all_reviewer_ratings.items()):
        avg_rating = np.mean(ratings)
        std_error = np.std(ratings) / np.sqrt(len(ratings))
        print(f"  {reviewer:3s}: {avg_rating:6.2f} ± {std_error:5.2f} (n={len(ratings):3d})")

if __name__ == "__main__":
    main()
