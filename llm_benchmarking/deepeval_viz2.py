#!/usr/bin/env python3
"""
DeepEval LLM v2 Results Visualization

This script creates comparison visualizations between:
1. LLM v1 results (original pipeline)
2. Reviewer vs Reviewer results 
3. LLM v2 results (new pipeline)

For the metrics: Faithfulness, Completeness, and Accuracy

Outputs:
- Individual metric plots by question type comparing all three approaches
- Overall comparison plot showing average scores across all approaches
"""

import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('dark_background')
sns.set_palette("husl")

# Define custom color palette for the three approaches
COMPARISON_COLORS = ['#FFB366', '#B19CD9', '#66B2FF']  # Orange, Lilac, Blue
APPROACH_LABELS = ['LLM v1', 'Reviewer vs Reviewer', 'LLM v2']

class DeepEvalLLMv2Visualizer:
    def __init__(self):
        """Initialize the DeepEval LLM v2 Visualizer."""
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set paths relative to the script location
        self.results_dir = os.path.join(script_dir, "deepeval-results")
        self.output_dir = os.path.join(script_dir, "deepeval-analyses")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define the three metrics we're comparing
        self.target_metrics = ['Faithfulness', 'Completeness', 'Accuracy']
        
        # Define question types
        self.question_types = [
            'additional_stressors', 'bee_species', 'experimental_methodology',
            'future_research', 'pesticides', 'significance'
        ]
        
        print(f"🎨 DeepEval LLM v2 Visualizer initialized")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"📁 Output directory: {self.output_dir}")

    def load_llm_v1_results(self) -> Dict[str, List[Dict]]:
        """Load LLM v1 results (original pipeline using Faithfulness, ContextualPrecision, ContextualRecall)"""
        llm_v1_results = {}
        
        print("📊 Loading LLM v1 results...")
        
        # Look for deepeval_results_* files (original pipeline)
        pattern = os.path.join(self.results_dir, "deepeval_results_*.json")
        files = glob.glob(pattern)
        
        for file_path in files:
            filename = os.path.basename(file_path)
            # Extract question type from filename
            question_type = filename.replace('deepeval_results_', '').split('_')[0]
            
            if question_type in self.question_types:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        llm_v1_results[question_type] = data
                        print(f"  ✅ Loaded LLM v1 {question_type}: {len(data)} test cases")
                except Exception as e:
                    print(f"  ⚠️  Error loading {filename}: {e}")
        
        return llm_v1_results

    def load_reviewer_vs_reviewer_results(self) -> List[Dict]:
        """Load reviewer vs reviewer results"""
        print("📊 Loading Reviewer vs Reviewer results...")
        
        # Look for deepeval_reviewer_results_* files
        pattern = os.path.join(self.results_dir, "deepeval_reviewer_results_*.json")
        files = glob.glob(pattern)
        
        if not files:
            print("  ⚠️  No reviewer vs reviewer results found")
            return []
        
        # Use the most recent file
        latest_file = max(files, key=os.path.getctime)
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
                print(f"  ✅ Loaded Reviewer vs Reviewer: {len(data)} test cases")
                return data
        except Exception as e:
            print(f"  ⚠️  Error loading reviewer results: {e}")
            return []

    def load_llm_v2_results(self) -> Dict[str, List[Dict]]:
        """Load LLM v2 results (new pipeline)"""
        llm_v2_results = {}
        
        print("📊 Loading LLM v2 results...")
        
        # Look for deepeval_llmv2_results_* files
        pattern = os.path.join(self.results_dir, "deepeval_llmv2_results_*.json")
        files = glob.glob(pattern)
        
        for file_path in files:
            filename = os.path.basename(file_path)
            # Extract question type from filename
            parts = filename.replace('deepeval_llmv2_results_', '').split('_')
            question_type = parts[0]
            
            if question_type in self.question_types:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        llm_v2_results[question_type] = data
                        print(f"  ✅ Loaded LLM v2 {question_type}: {len(data)} test cases")
                except Exception as e:
                    print(f"  ⚠️  Error loading {filename}: {e}")
        
        return llm_v2_results

    def extract_metric_scores(self, results: List[Dict], target_metrics: List[str]) -> Dict[str, List[float]]:
        """Extract scores for target metrics from results"""
        metric_scores = {metric: [] for metric in target_metrics}
        
        for result in results:
            if 'metrics_data' in result and result['metrics_data']:
                for metric_data in result['metrics_data']:
                    metric_name = metric_data.get('name', '')
                    score = metric_data.get('score')
                    
                    # Handle different metric name formats
                    for target_metric in target_metrics:
                        # More flexible matching for metric names
                        if (target_metric.lower() in metric_name.lower() or 
                            metric_name.lower().startswith(target_metric.lower()) or
                            target_metric.lower() in metric_name.lower().replace('[geval]', '').replace('metric', '').strip()):
                            if score is not None:
                                metric_scores[target_metric].append(score)
                                print(f"  🔍 Matched '{metric_name}' -> '{target_metric}': {score:.3f}")
                            break
        
        return metric_scores

    def prepare_comparison_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for comparison visualizations"""
        print("🔄 Preparing comparison data...")
        
        # Load all results
        llm_v1_results = self.load_llm_v1_results()
        reviewer_results = self.load_reviewer_vs_reviewer_results()
        llm_v2_results = self.load_llm_v2_results()
        
        # Prepare detailed data (by question type)
        detailed_data = []
        
        # Prepare summary data (averages)
        summary_data = []
        
        # Process LLM v1 results
        for question_type, results in llm_v1_results.items():
            metric_scores = self.extract_metric_scores(results, self.target_metrics)
            
            for metric, scores in metric_scores.items():
                if scores:  # Only if we have scores
                    avg_score = np.mean(scores)
                    
                    # Add to detailed data
                    for score in scores:
                        detailed_data.append({
                            'Question': question_type.replace('_', ' ').title(),
                            'Approach': 'LLM v1',
                            'Metric': metric,
                            'Score': score
                        })
                    
                    # Add to summary data
                    summary_data.append({
                        'Approach': 'LLM v1',
                        'Metric': metric,
                        'Average_Score': avg_score,
                        'Count': len(scores)
                    })
        
        # Process Reviewer vs Reviewer results
        if reviewer_results:
            # Group by question type
            reviewer_by_question = {}
            for result in reviewer_results:
                if 'additional_metadata' in result:
                    question_id = result['additional_metadata'].get('question_id', 'unknown')
                    if question_id not in reviewer_by_question:
                        reviewer_by_question[question_id] = []
                    reviewer_by_question[question_id].append(result)
            
            for question_type, results in reviewer_by_question.items():
                if question_type in self.question_types:
                    metric_scores = self.extract_metric_scores(results, self.target_metrics)
                    
                    for metric, scores in metric_scores.items():
                        if scores:  # Only if we have scores
                            avg_score = np.mean(scores)
                            
                            # Add to detailed data
                            for score in scores:
                                detailed_data.append({
                                    'Question': question_type.replace('_', ' ').title(),
                                    'Approach': 'Reviewer vs Reviewer',
                                    'Metric': metric,
                                    'Score': score
                                })
                            
                            # Add to summary data
                            summary_data.append({
                                'Approach': 'Reviewer vs Reviewer',
                                'Metric': metric,
                                'Average_Score': avg_score,
                                'Count': len(scores)
                            })
        
        # Process LLM v2 results
        for question_type, results in llm_v2_results.items():
            metric_scores = self.extract_metric_scores(results, self.target_metrics)
            
            for metric, scores in metric_scores.items():
                if scores:  # Only if we have scores
                    avg_score = np.mean(scores)
                    
                    # Add to detailed data
                    for score in scores:
                        detailed_data.append({
                            'Question': question_type.replace('_', ' ').title(),
                            'Approach': 'LLM v2',
                            'Metric': metric,
                            'Score': score
                        })
                    
                    # Add to summary data
                    summary_data.append({
                        'Approach': 'LLM v2',
                        'Metric': metric,
                        'Average_Score': avg_score,
                        'Count': len(scores)
                    })
        
        detailed_df = pd.DataFrame(detailed_data)
        summary_df = pd.DataFrame(summary_data)
        
        print(f"✅ Prepared {len(detailed_df)} detailed records and {len(summary_df)} summary records")
        
        return detailed_df, summary_df

    def create_metric_by_question_plots(self, detailed_df: pd.DataFrame):
        """Create individual plots for each metric showing performance by question type"""
        print("🎨 Creating metric by question plots...")
        
        for metric in self.target_metrics:
            metric_data = detailed_df[detailed_df['Metric'] == metric]
            
            if metric_data.empty:
                print(f"  ⚠️  No data found for {metric}")
                continue
            
            plt.figure(figsize=(14, 8))
            
            # Create box plot
            ax = sns.boxplot(
                data=metric_data,
                x='Question',
                y='Score',
                hue='Approach',
                palette=COMPARISON_COLORS
            )
            
            plt.title(f'{metric} Scores by Question Type\nComparison: LLM v1 vs Reviewer vs Reviewer vs LLM v2', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Question Type', fontsize=12, fontweight='bold')
            plt.ylabel(f'{metric} Score', fontsize=12, fontweight='bold')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Customize legend
            plt.legend(title='Approach', title_fontsize=12, fontsize=10, 
                      bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add grid for better readability
            plt.grid(True, alpha=0.3, axis='y')
            
            # Set y-axis to 0-1 range
            plt.ylim(0, 1)
            
            plt.tight_layout()
            
            # Save the plot
            filename = f"{metric}_by_question_type_comparison.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='#2E2E2E')
            
            # Also save as PDF
            pdf_filepath = os.path.join(self.output_dir, filename.replace('.png', '.pdf'))
            plt.savefig(pdf_filepath, dpi=300, bbox_inches='tight', facecolor='#2E2E2E')
            
            plt.close()
            
            print(f"  ✅ Saved {metric} comparison plot: {filename}")

    def create_overall_comparison_plot(self, summary_df: pd.DataFrame):
        """Create overall comparison plot showing average scores across all approaches"""
        print("🎨 Creating overall comparison plot...")
        
        if summary_df.empty:
            print("  ⚠️  No summary data available")
            return
        
        # Calculate overall averages by approach and metric
        overall_avg = summary_df.groupby(['Approach', 'Metric'])['Average_Score'].mean().reset_index()
        
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar plot
        ax = sns.barplot(
            data=overall_avg,
            x='Metric',
            y='Average_Score',
            hue='Approach',
            palette=COMPARISON_COLORS
        )
        
        plt.title('Average Performance Comparison\nLLM v1 vs Reviewer vs Reviewer vs LLM v2', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Metric', fontsize=12, fontweight='bold')
        plt.ylabel('Average Score', fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=10)
        
        # Customize legend
        plt.legend(title='Approach', title_fontsize=12, fontsize=10, 
                  bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis to 0-1 range
        plt.ylim(0, 1)
        
        plt.tight_layout()
        
        # Save the plot
        filename = "overall_performance_comparison.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='#2E2E2E')
        
        # Also save as PDF
        pdf_filepath = os.path.join(self.output_dir, filename.replace('.png', '.pdf'))
        plt.savefig(pdf_filepath, dpi=300, bbox_inches='tight', facecolor='#2E2E2E')
        
        plt.close()
        
        print(f"  ✅ Saved overall comparison plot: {filename}")
        
        # Print summary statistics
        print("\n📊 Summary Statistics:")
        for approach in ['LLM v1', 'Reviewer vs Reviewer', 'LLM v2']:
            approach_data = overall_avg[overall_avg['Approach'] == approach]
            if not approach_data.empty:
                avg_performance = approach_data['Average_Score'].mean()
                print(f"  {approach}: {avg_performance:.3f} average across all metrics")

    def save_summary_data(self, detailed_df: pd.DataFrame, summary_df: pd.DataFrame):
        """Save summary data to JSON files"""
        print("💾 Saving summary data...")
        
        # Save detailed data
        detailed_file = os.path.join(self.output_dir, "llmv2_detailed_comparison_data.json")
        detailed_df.to_json(detailed_file, orient='records', indent=2)
        
        # Save summary data
        summary_file = os.path.join(self.output_dir, "llmv2_summary_comparison_data.json")
        summary_df.to_json(summary_file, orient='records', indent=2)
        
        print(f"  ✅ Saved detailed data: llmv2_detailed_comparison_data.json")
        print(f"  ✅ Saved summary data: llmv2_summary_comparison_data.json")

    def run_analysis(self):
        """Run the complete analysis and generate visualizations"""
        print("🚀 Starting LLM v2 comparison analysis...")
        
        try:
            # Prepare data
            detailed_df, summary_df = self.prepare_comparison_data()
            
            if detailed_df.empty:
                print("❌ No data available for analysis")
                return
            
            # Create visualizations
            self.create_metric_by_question_plots(detailed_df)
            self.create_overall_comparison_plot(summary_df)
            
            # Save data
            self.save_summary_data(detailed_df, summary_df)
            
            print(f"\n🎉 Analysis complete! Results saved in: {self.output_dir}")
            
            # List all created files
            output_files = [f for f in os.listdir(self.output_dir) if f.startswith(('Faithfulness', 'Completeness', 'Accuracy', 'overall', 'llmv2'))]
            print(f"📁 Created {len(output_files)} output files:")
            for file in sorted(output_files):
                print(f"  - {file}")
                
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            raise

def main():
    """Main function to run the visualization"""
    visualizer = DeepEvalLLMv2Visualizer()
    visualizer.run_analysis()

if __name__ == "__main__":
    main()
