#!/usr/bin/env python3
"""
Example plotting script using the improved DeepEval results dataframe.

This script demonstrates how to use the improved summary_all_results_improved.csv
for creating various plots and analyses.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results_data():
    """Load the improved results data."""
    script_dir = Path(__file__).parent
    results_file = script_dir / "deepeval-results" / "summary_all_results_improved.csv"
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None
    
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} records from {results_file}")
    return df


def plot_metric_comparison_by_type(df, metric_name="Faithfulness"):
    """
    Plot a specific metric across different comparison types.
    
    Args:
        df: The results dataframe
        metric_name: The metric to plot
    """
    # Filter for the specific metric
    metric_data = df[df['metric'] == metric_name].copy()
    
    if metric_data.empty:
        print(f"No data found for metric: {metric_name}")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create a grouped bar plot
    sns.barplot(data=metric_data, x='question_type', y='average', 
                hue='comparison_label', palette='Set2')
    
    plt.title(f'{metric_name} Scores by Question Type and Comparison Type', fontsize=14)
    plt.xlabel('Question Type', fontsize=12)
    plt.ylabel(f'{metric_name} Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Comparison Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    output_file = f'{metric_name.lower().replace(" ", "_")}_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_file}")
    plt.show()


def plot_question_type_performance(df, question_type="bee_species"):
    """
    Plot performance across all metrics for a specific question type.
    
    Args:
        df: The results dataframe
        question_type: The question type to analyze
    """
    # Filter for the specific question type
    question_data = df[df['question_type'] == question_type].copy()
    
    if question_data.empty:
        print(f"No data found for question type: {question_type}")
        return
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Create a grouped bar plot
    sns.barplot(data=question_data, x='metric', y='average', 
                hue='comparison_label', palette='Set1')
    
    plt.title(f'Performance Across Metrics for {question_type.replace("_", " ").title()}', fontsize=14)
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Comparison Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    output_file = f'{question_type}_performance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_file}")
    plt.show()


def create_heatmap(df, metric_name="Faithfulness"):
    """
    Create a heatmap showing performance across question types and comparison types.
    
    Args:
        df: The results dataframe
        metric_name: The metric to visualize
    """
    # Filter for the specific metric
    metric_data = df[df['metric'] == metric_name].copy()
    
    if metric_data.empty:
        print(f"No data found for metric: {metric_name}")
        return
    
    # Pivot the data for heatmap
    pivot_data = metric_data.pivot_table(
        index='question_type', 
        columns='comparison_label', 
        values='average', 
        aggfunc='mean'
    )
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.3f', 
                cbar_kws={'label': f'{metric_name} Score'})
    
    plt.title(f'{metric_name} Performance Heatmap', fontsize=14)
    plt.xlabel('Comparison Type', fontsize=12)
    plt.ylabel('Question Type', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    output_file = f'{metric_name.lower().replace(" ", "_")}_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved as: {output_file}")
    plt.show()


def create_summary_table(df):
    """
    Create a summary table showing overall performance by comparison type.
    
    Args:
        df: The results dataframe
    """
    # Calculate summary statistics
    summary = df.groupby(['comparison_type', 'metric']).agg({
        'average': ['mean', 'std', 'count'],
        'count': 'sum'
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    print("\nSummary Statistics by Comparison Type and Metric:")
    print("=" * 60)
    print(summary.to_string(index=False))
    
    return summary


def main():
    """Main function to demonstrate plotting capabilities."""
    print("DeepEval Results Plotting Example")
    print("=" * 40)
    
    # Load the data
    df = load_results_data()
    if df is None:
        return
    
    # Display basic info about the data
    print(f"\nData Overview:")
    print(f"- Total records: {len(df)}")
    print(f"- Comparison types: {df['comparison_type'].nunique()}")
    print(f"- Question types: {df['question_type'].nunique()}")
    print(f"- Metrics: {df['metric'].nunique()}")
    
    print(f"\nAvailable comparison types: {list(df['comparison_type'].unique())}")
    print(f"Available question types: {list(df['question_type'].unique())}")
    print(f"Available metrics: {list(df['metric'].unique())}")
    
    # Create summary table
    summary = create_summary_table(df)
    
    # Example plots (uncomment to generate)
    print("\nGenerating example plots...")
    
    # Plot 1: Faithfulness comparison
    plot_metric_comparison_by_type(df, "Faithfulness")
    
    # Plot 2: Bee species performance
    plot_question_type_performance(df, "bee_species")
    
    # Plot 3: Heatmap
    create_heatmap(df, "Faithfulness")
    
    print("\nPlotting example complete!")
    print("\nThe improved dataframe structure allows for:")
    print("- Easy filtering by comparison_type, question_type, or metric")
    print("- Grouped plotting with comparison_label for clear legends")
    print("- Heatmap visualization of performance patterns")
    print("- Statistical analysis across different dimensions")


if __name__ == "__main__":
    main()
