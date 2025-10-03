#!/usr/bin/env python3
"""
Create bar graphs with error bars comparing performance across comparison types.

This script generates bar plots for each question type and metric, showing performance
across the three main comparison types: llmv1_vs_rev1, rev1_vs_rev2, and llmv2_vs_rev3.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os


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


# Removed individual bar plot function - we only need grid plots


# Removed individual plots function - we only need grid plots


def create_metric_comparison_grid(df, metric, save_dir="plots"):
    """
    Create a grid of bar plots for a specific metric across all question types.
    
    Args:
        df: The results dataframe
        metric: The metric to plot
        save_dir: Directory to save plots
    """
    # Filter data for the specific metric
    metric_data = df[df['metric'] == metric].copy()
    
    if metric_data.empty:
        print(f"No data found for metric: {metric}")
        return
    
    # Get unique question types
    question_types = metric_data['question_type'].unique()
    n_questions = len(question_types)
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_questions + n_cols - 1) // n_cols
    
    # Create the subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Define comparison types based on metric
    if metric in ['Contextual Precision', 'Contextual Recall']:
        comparison_types = ['llmv1_vs_rev1', 'llmv2_vs_rev3']  # Exclude rev1_vs_rev2
        colors = ['#1f77b4', '#2ca02c']
    else:
        comparison_types = ['llmv1_vs_rev1', 'rev1_vs_rev2', 'llmv2_vs_rev3']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, question_type in enumerate(question_types):
        ax = axes_flat[i]
        
        # Filter data for this question type
        plot_data = metric_data[metric_data['question_type'] == question_type]
        plot_data = plot_data[plot_data['comparison'].isin(comparison_types)]
        
        if plot_data.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{question_type.replace("_", " ").title()}', fontweight='bold')
            continue
        
        # Get data for each comparison type
        averages = []
        errors = []
        labels = []
        
        for comp_type in comparison_types:
            comp_data = plot_data[plot_data['comparison'] == comp_type]
            if not comp_data.empty:
                avg = comp_data['average'].iloc[0]
                std_err = comp_data['standard_error'].iloc[0]
                averages.append(avg)
                errors.append(std_err)
                labels.append(comp_type.replace('_', ' ').title())
            else:
                averages.append(0)
                errors.append(0)
                labels.append(comp_type.replace('_', ' ').title())
        
        # Create the bar plot
        x_pos = np.arange(len(comparison_types))
        bars = ax.bar(x_pos, averages, yerr=errors, capsize=3, 
                      color=colors, alpha=0.8, edgecolor='black')
        
        # Customize subplot
        ax.set_title(f'{question_type.replace("_", " ").title()}', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, avg, err in zip(bars, averages, errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.02,
                   f'{avg:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Hide empty subplots
    for i in range(len(question_types), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'{metric} Performance Across Question Types', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot as PDF
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{metric.lower().replace(' ', '_').replace('[', '').replace(']', '')}_grid.pdf"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    print(f"Saved grid plot: {filepath}")
    plt.close()  # Close the figure instead of showing it


def create_summary_statistics_table(df, save_dir="plots"):
    """
    Create a summary statistics table and save as CSV.
    
    Args:
        df: The results dataframe
        save_dir: Directory to save files
    """
    # Filter to only include the three main comparison types
    comparison_types = ['llmv1_vs_rev1', 'rev1_vs_rev2', 'llmv2_vs_rev3']
    df_filtered = df[df['comparison'].isin(comparison_types)]
    
    # Create summary table
    summary = df_filtered.groupby(['comparison', 'question_type', 'metric']).agg({
        'average': ['mean', 'std', 'min', 'max'],
        'count': 'sum',
        'standard_error': 'mean'
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    # Save summary table
    os.makedirs(save_dir, exist_ok=True)
    summary_file = os.path.join(save_dir, "summary_statistics.csv")
    summary.to_csv(summary_file, index=False)
    print(f"Summary statistics saved: {summary_file}")
    
    return summary


def create_condensed_horizontal_plots(df, save_dir="plots"):
    """
    Create condensed horizontal bar plots for each metric with question types on y-axis.
    
    Args:
        df: The results dataframe
        save_dir: Directory to save plots
    """
    # Filter to only include the three main comparison types
    comparison_types = ['llmv1_vs_rev1', 'rev1_vs_rev2', 'llmv2_vs_rev3']
    df_filtered = df[df['comparison'].isin(comparison_types)]
    
    # Get unique metrics
    metrics = df_filtered['metric'].unique()
    
    # Define comparison types and colors for each metric
    for metric in metrics:
        # Define comparison types based on metric (ordered: LLM v1 top, Rev1 middle, LLM v2 bottom)
        if metric in ['Contextual Precision', 'Contextual Recall']:
            comp_types = ['llmv1_vs_rev1', 'llmv2_vs_rev3']  # Exclude rev1_vs_rev2
            colors = ['#1f77b4', '#2ca02c']
            labels = ['LLM v1', 'LLM v2']
        else:
            comp_types = ['llmv1_vs_rev1', 'rev1_vs_rev2', 'llmv2_vs_rev3']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            labels = ['LLM v1', 'Rev1', 'LLM v2']
        
        # Get data for this metric
        metric_data = df_filtered[df_filtered['metric'] == metric]
        
        if metric_data.empty:
            continue
        
        # Get only the specified question types
        allowed_question_types = ['pesticides', 'bee_species', 'additional_stressors']
        question_types = [q for q in metric_data['question_type'].unique() if q in allowed_question_types]
        n_questions = len(question_types)
        
        # Create the plot (less wide, same height)
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Set up the bar positions
        y_pos = np.arange(n_questions)
        bar_width = 0.25
        
        # Plot bars for each comparison type (reverse order so LLM v1 is at top)
        for i, (comp_type, color, label) in enumerate(zip(comp_types, colors, labels)):
            averages = []
            errors = []
            
            for question_type in question_types:
                comp_data = metric_data[(metric_data['comparison'] == comp_type) & 
                                      (metric_data['question_type'] == question_type)]
                
                if not comp_data.empty:
                    avg = comp_data['average'].iloc[0]
                    std_err = comp_data['standard_error'].iloc[0]
                    averages.append(avg)
                    errors.append(std_err)
                else:
                    averages.append(0)
                    errors.append(0)
            
            # Create horizontal bars (reverse i to put LLM v1 at top, no error bars or labels)
            bars = ax.barh(y_pos + (len(comp_types) - 1 - i) * bar_width, averages, bar_width, 
                          color=color, alpha=0.8, edgecolor='black', label=label)
        
        # Customize the plot
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Score', fontsize=10)
        ax.set_ylabel('Question Type', fontsize=10)
        ax.set_yticks(y_pos + bar_width)
        ax.set_yticklabels([q.replace('_', ' ').title() for q in question_types], fontsize=9)
        ax.set_xlim(0, 1.1)
        ax.grid(axis='x', alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
        
        plt.tight_layout()
        
        # Save the plot as PDF
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{metric.lower().replace(' ', '_').replace('[', '').replace(']', '')}_horizontal.pdf"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, format='pdf', bbox_inches='tight')
        print(f"Saved horizontal plot: {filepath}")
        plt.close()  # Close the figure instead of showing it


def create_overall_average_grid(df, save_dir="plots"):
    """
    Create a grid plot showing average performance across all question types for each metric.
    
    Args:
        df: The results dataframe
        save_dir: Directory to save plots
    """
    # Get unique metrics
    metrics = df['metric'].unique()
    n_metrics = len(metrics)
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create the subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes_flat[i]
        
        # Define comparison types based on metric
        if metric in ['Contextual Precision', 'Contextual Recall']:
            comparison_types = ['llmv1_vs_rev1', 'llmv2_vs_rev3']  # Exclude rev1_vs_rev2
            colors = ['#1f77b4', '#2ca02c']
        else:
            comparison_types = ['llmv1_vs_rev1', 'rev1_vs_rev2', 'llmv2_vs_rev3']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # Calculate average across all question types for each comparison type
        averages = []
        errors = []
        labels = []
        
        for comp_type in comparison_types:
            # Get data for this comparison type and metric
            comp_data = df[(df['comparison'] == comp_type) & (df['metric'] == metric)]
            
            if not comp_data.empty:
                # Calculate weighted average (weighted by count) and combined standard error
                total_count = comp_data['count'].sum()
                weighted_avg = (comp_data['average'] * comp_data['count']).sum() / total_count
                
                # Calculate combined standard error
                # Using the formula for combining standard errors of means
                combined_var = (comp_data['standard_error'] ** 2 * comp_data['count']).sum() / total_count
                combined_std_err = np.sqrt(combined_var)
                
                averages.append(weighted_avg)
                errors.append(combined_std_err)
                labels.append(comp_type.replace('_', ' ').title())
            else:
                averages.append(0)
                errors.append(0)
                labels.append(comp_type.replace('_', ' ').title())
        
        # Create the bar plot
        x_pos = np.arange(len(comparison_types))
        bars = ax.bar(x_pos, averages, yerr=errors, capsize=3, 
                      color=colors, alpha=0.8, edgecolor='black')
        
        # Customize subplot
        ax.set_title(f'{metric}', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, avg, err in zip(bars, averages, errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.02,
                   f'{avg:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Hide empty subplots
    for i in range(len(metrics), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Add overall title
    fig.suptitle('Overall Average Performance Across All Question Types', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot as PDF
    os.makedirs(save_dir, exist_ok=True)
    filename = "overall_average_performance_grid.pdf"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    print(f"Saved overall average grid plot: {filepath}")
    plt.close()  # Close the figure instead of showing it


def main():
    """Main function to create all comparison plots."""
    print("Creating Comparison Bar Plots with Error Bars")
    print("=" * 50)
    
    # Load the data
    df = load_results_data()
    if df is None:
        return
    
    # Create output directory in the same folder as the script
    script_dir = Path(__file__).parent
    save_dir = script_dir / "comparison_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # Display data overview
    print(f"\nData Overview:")
    print(f"- Total records: {len(df)}")
    print(f"- Question types: {list(df['question_type'].unique())}")
    print(f"- Metrics: {list(df['metric'].unique())}")
    print(f"- Comparison types: {list(df['comparison'].unique())}")
    
    # Create summary statistics table
    print("\nCreating summary statistics table...")
    summary = create_summary_statistics_table(df, save_dir)
    
    # Create grid plots for each metric
    print("\nCreating grid plots for each metric...")
    metrics = df['metric'].unique()
    for metric in metrics:
        print(f"Creating grid plot for: {metric}")
        create_metric_comparison_grid(df, metric, save_dir)
    
    # Create overall average grid plot
    print("\nCreating overall average performance grid...")
    create_overall_average_grid(df, save_dir)
    
    # Create condensed horizontal plots
    print("\nCreating condensed horizontal plots...")
    create_condensed_horizontal_plots(df, save_dir)
    
    print(f"\nAll plots created and saved in: {save_dir}/")
    print("\nPlot types created:")
    print("- Grid plots showing all question types for each metric (saved as PDFs)")
    print("- Overall average performance grid across all question types (saved as PDF)")
    print("- Condensed horizontal plots for each metric (saved as PDFs)")
    print("- Summary statistics table")


if __name__ == "__main__":
    main()
