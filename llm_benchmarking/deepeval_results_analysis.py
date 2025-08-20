#!/usr/bin/env python3
"""
DeepEval Results Analysis

This script analyzes and visualizes DeepEval evaluation results by:
1. Merging results from deepeval-results/ into consolidated files
2. Creating comparison plots between LLM vs reviewer and reviewer vs reviewer data
3. Generating individual metric plots for detailed analysis

Outputs:
- Merged data files (JSON/JSONL) without context fields
- Bar plots comparing average scores across metrics
- Individual metric plots grouped by question type
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
plt.style.use('default')
sns.set_palette("husl")

class DeepEvalResultsAnalyzer:
    def __init__(self):
        """Initialize the DeepEval Results Analyzer."""
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set paths relative to the script location
        self.results_dir = os.path.join(script_dir, "deepeval-results")
        self.output_dir = os.path.join(script_dir, "deepeval-analyses")
        self.merged_data_dir = os.path.join(self.output_dir, "merged-data")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.merged_data_dir, exist_ok=True)
        
        # Define metric categories
        self.llm_metrics = [
            "FaithfulnessMetric", 
            "ContextualPrecisionMetric", 
            "ContextualRecallMetric"
        ]
        
        self.geval_metrics = [
            "Correctness", 
            "Completeness", 
            "Accuracy"
        ]
        
        print(f"✅ Initialized analyzer")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📁 Merged data directory: {self.merged_data_dir}")
    
    def load_results_files(self) -> Tuple[List[Dict], List[Dict]]:
        """Load all DeepEval results files and separate LLM vs reviewer and reviewer vs reviewer data."""
        print("\n🔄 Loading DeepEval results files...")
        
        # Find all JSON result files
        pattern = os.path.join(self.results_dir, "*.json")
        result_files = glob.glob(pattern)
        
        if not result_files:
            raise FileNotFoundError(f"No result files found in {self.results_dir}")
        
        print(f"📁 Found {len(result_files)} result files")
        
        llm_results = []
        reviewer_results = []
        
        for file_path in result_files:
            filename = os.path.basename(file_path)
            print(f"  📄 Processing: {filename}")
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Determine if this is LLM vs reviewer or reviewer vs reviewer
                if "reviewer" in filename.lower() or "reviewer" in filename:
                    reviewer_results.extend(data)
                    print(f"    → Added to reviewer comparison data ({len(data)} entries)")
                else:
                    llm_results.extend(data)
                    print(f"    → Added to LLM comparison data ({len(data)} entries)")
                    
            except Exception as e:
                print(f"    ⚠️  Error loading {filename}: {e}")
                continue
        
        print(f"\n📊 Data loaded:")
        print(f"  LLM vs Reviewer: {len(llm_results)} entries")
        print(f"  Reviewer vs Reviewer: {len(reviewer_results)} entries")
        
        return llm_results, reviewer_results
    
    def clean_and_merge_results(self, results: List[Dict], data_type: str) -> List[Dict]:
        """Clean and merge results, removing context fields and standardizing structure."""
        print(f"\n🧹 Cleaning and merging {data_type} results...")
        
        cleaned_results = []
        
        for entry in results:
            try:
                # Create cleaned entry without context fields
                cleaned_entry = {
                    "test_case_index": entry.get("test_case_index"),
                    "name": entry.get("name"),
                    "input": entry.get("input"),
                    "actual_output": entry.get("actual_output"),
                    "expected_output": entry.get("expected_output"),
                    "success": entry.get("success"),
                    "additional_metadata": entry.get("additional_metadata"),
                    "metrics_data": []
                }
                
                # Clean metrics data
                if entry.get("metrics_data"):
                    for metric in entry["metrics_data"]:
                        cleaned_metric = {
                            "name": metric.get("name"),
                            "score": metric.get("score"),
                            "threshold": metric.get("threshold"),
                            "success": metric.get("success"),
                            "reason": metric.get("reason"),
                            "strict_mode": metric.get("strict_mode"),
                            "evaluation_model": metric.get("evaluation_model"),
                            "error": metric.get("error"),
                            "evaluation_cost": metric.get("evaluation_cost")
                        }
                        cleaned_entry["metrics_data"].append(cleaned_metric)
                
                cleaned_results.append(cleaned_entry)
                
            except Exception as e:
                print(f"    ⚠️  Error cleaning entry: {e}")
                continue
        
        print(f"  ✅ Cleaned {len(cleaned_results)} entries")
        return cleaned_results
    
    def save_merged_data(self, llm_results: List[Dict], reviewer_results: List[Dict]):
        """Save merged data to JSON and JSONL files."""
        print(f"\n💾 Saving merged data...")
        
        # Save LLM comparison results
        llm_json_path = os.path.join(self.merged_data_dir, "merged_llm_comparison_results.json")
        llm_jsonl_path = os.path.join(self.merged_data_dir, "merged_llm_comparison_results.jsonl")
        
        with open(llm_json_path, 'w') as f:
            json.dump(llm_results, f, indent=2)
        
        with open(llm_jsonl_path, 'w') as f:
            for entry in llm_results:
                f.write(json.dumps(entry) + '\n')
        
        # Save reviewer comparison results
        reviewer_json_path = os.path.join(self.merged_data_dir, "merged_reviewer_comparison_results.json")
        reviewer_jsonl_path = os.path.join(self.merged_data_dir, "merged_reviewer_comparison_results.jsonl")
        
        with open(reviewer_json_path, 'w') as f:
            json.dump(reviewer_results, f, indent=2)
        
        with open(reviewer_jsonl_path, 'w') as f:
            for entry in reviewer_results:
                f.write(json.dumps(entry) + '\n')
        
        print(f"  ✅ LLM comparison: {llm_json_path}")
        print(f"  ✅ LLM comparison: {llm_jsonl_path}")
        print(f"  ✅ Reviewer comparison: {reviewer_json_path}")
        print(f"  ✅ Reviewer comparison: {reviewer_jsonl_path}")
    
    def extract_metric_data(self, results: List[Dict], data_type: str) -> pd.DataFrame:
        """Extract metric data into a pandas DataFrame for analysis."""
        print(f"\n📊 Extracting metric data from {data_type} results...")
        
        metric_data = []
        
        for entry in results:
            if not entry.get("metrics_data"):
                continue
                
            for metric in entry["metrics_data"]:
                metric_name = metric.get("name", "Unknown")
                score = metric.get("score")
                
                if score is not None:
                    row = {
                        "data_type": data_type,
                        "metric_name": metric_name,
                        "score": score,
                        "success": metric.get("success", False),
                        "test_case_index": entry.get("test_case_index"),
                        "input": entry.get("input", ""),
                        "question_type": self._extract_question_type(entry),
                        "paper_id": self._extract_paper_id(entry)
                    }
                    metric_data.append(row)
        
        df = pd.DataFrame(metric_data)
        print(f"  ✅ Extracted {len(df)} metric measurements")
        if len(df) > 0:
            print(f"  📈 Metrics found: {df['metric_name'].unique()}")
        else:
            print(f"  📈 No metrics found (empty dataset)")
        
        return df
    
    def _extract_question_type(self, entry: Dict) -> str:
        """Extract question type from entry metadata or input."""
        if entry.get("additional_metadata"):
            metadata = entry["additional_metadata"]
            if isinstance(metadata, dict):
                return metadata.get("question_id", "unknown")
            elif isinstance(metadata, str):
                try:
                    parsed = json.loads(metadata)
                    return parsed.get("question_id", "unknown")
                except:
                    pass
        
        # Fallback: try to extract from input
        input_text = entry.get("input", "")
        if "bee species" in input_text.lower():
            return "bee_species"
        elif "pesticide" in input_text.lower():
            return "pesticides"
        elif "stressor" in input_text.lower():
            return "additional_stressors"
        elif "methodology" in input_text.lower():
            return "experimental_methodology"
        elif "significance" in input_text.lower():
            return "significance"
        elif "future research" in input_text.lower():
            return "future_research"
        elif "limitation" in input_text.lower():
            return "limitations"
        
        return "unknown"
    
    def _extract_paper_id(self, entry: Dict) -> str:
        """Extract paper ID from entry metadata."""
        if entry.get("additional_metadata"):
            metadata = entry["additional_metadata"]
            if isinstance(metadata, dict):
                return metadata.get("paper_id", "unknown")
            elif isinstance(metadata, str):
                try:
                    parsed = json.loads(metadata)
                    return parsed.get("paper_id", "unknown")
                except:
                    pass
        
        return "unknown"
    
    def create_comparison_plots(self, llm_df: pd.DataFrame, reviewer_df: pd.DataFrame):
        """Create comparison plots between LLM vs reviewer and reviewer vs reviewer data."""
        print(f"\n📊 Creating comparison plots...")
        
        # Combine data for plotting
        combined_df = pd.concat([llm_df, reviewer_df], ignore_index=True)
        
        # Create average scores plot
        self._create_average_scores_plot(combined_df)
        
        # Create individual metric plots
        self._create_individual_metric_plots(combined_df)
        
        print(f"  ✅ All plots created and saved to {self.output_dir}")
    
    def _create_average_scores_plot(self, df: pd.DataFrame):
        """Create bar plot comparing average scores across metrics."""
        print(f"  📈 Creating average scores comparison plot...")
        
        # Calculate averages and standard errors for each metric
        plot_data = []
        
        for metric_name in df['metric_name'].unique():
            for data_type in df['data_type'].unique():
                subset = df[(df['metric_name'] == metric_name) & (df['data_type'] == data_type)]
                
                if len(subset) > 0:
                    if metric_name == "Correctness":
                        # For Correctness, calculate proportion of successful evaluations
                        success_rate = subset['success'].mean()
                        std_error = np.sqrt(success_rate * (1 - success_rate) / len(subset))
                        plot_data.append({
                            'metric_name': metric_name,
                            'data_type': data_type,
                            'value': success_rate,
                            'std_error': std_error,
                            'is_binary': True
                        })
                    else:
                        # For other metrics, calculate mean and standard error
                        mean_score = subset['score'].mean()
                        std_error = subset['score'].std() / np.sqrt(len(subset))
                        plot_data.append({
                            'metric_name': metric_name,
                            'data_type': data_type,
                            'value': mean_score,
                            'std_error': std_error,
                            'is_binary': False
                        })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Set up the plot
        x = np.arange(len(plot_df['metric_name'].unique()))
        width = 0.35
        
        # Plot bars for each data type
        for i, data_type in enumerate(plot_df['data_type'].unique()):
            data_subset = plot_df[plot_df['data_type'] == data_type]
            values = data_subset['value'].values
            errors = data_subset['std_error'].values
            
            # Create labels for x-axis
            metric_labels = [f"{m}\n({'Binary' if plot_df[plot_df['metric_name'] == m]['is_binary'].iloc[0] else 'Continuous'})" 
                           for m in data_subset['metric_name'].values]
            
            if i == 0:
                x_pos = x - width/2
            else:
                x_pos = x + width/2
            
            bars = plt.bar(x_pos, values, width, 
                          label=data_type.replace('_', ' ').title(),
                          yerr=errors, capsize=5, alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score / Success Rate')
        plt.title('Average Scores Comparison: LLM vs Reviewer vs Reviewer vs Reviewer')
        plt.xticks(x, metric_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, "average_scores_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Saved: average_scores_comparison.png")
    
    def _create_individual_metric_plots(self, df: pd.DataFrame):
        """Create individual plots for each metric grouped by question type."""
        print(f"  📊 Creating individual metric plots...")
        
        # Get unique metrics
        metrics = df['metric_name'].unique()
        
        for metric_name in metrics:
            print(f"    📈 Creating plot for {metric_name}...")
            
            # Filter data for this metric
            metric_df = df[df['metric_name'] == metric_name].copy()
            
            if len(metric_df) == 0:
                continue
            
            # Create the plot
            plt.figure(figsize=(16, 10))
            
            # Get unique question types
            question_types = sorted(metric_df['question_type'].unique())
            
            # Set up the plot
            x = np.arange(len(question_types))
            width = 0.35
            
            # Plot bars for each data type
            for i, data_type in enumerate(metric_df['data_type'].unique()):
                data_subset = metric_df[metric_df['data_type'] == data_type]
                
                # Calculate means for each question type
                means = []
                errors = []
                
                for question_type in question_types:
                    subset = data_subset[data_subset['question_type'] == question_type]
                    
                    if len(subset) > 0:
                        if metric_name == "Correctness":
                            # For Correctness, calculate proportion
                            mean_val = subset['success'].mean()
                            error_val = np.sqrt(mean_val * (1 - mean_val) / len(subset))
                        else:
                            # For other metrics, calculate mean and standard error
                            mean_val = subset['score'].mean()
                            error_val = subset['score'].std() / np.sqrt(len(subset))
                        
                        means.append(mean_val)
                        errors.append(error_val)
                    else:
                        means.append(0)
                        errors.append(0)
                
                # Position bars
                if i == 0:
                    x_pos = x - width/2
                else:
                    x_pos = x + width/2
                
                bars = plt.bar(x_pos, means, width, 
                              label=data_type.replace('_', ' ').title(),
                              yerr=errors, capsize=5, alpha=0.8)
                
                # Add value labels on bars
                for bar, value in zip(bars, means):
                    if value > 0:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.xlabel('Question Types')
            if metric_name == "Correctness":
                plt.ylabel('Success Rate')
                plt.title(f'{metric_name}: Success Rate by Question Type')
            else:
                plt.ylabel('Score')
                plt.title(f'{metric_name}: Average Score by Question Type')
            
            plt.xticks(x, [q.replace('_', ' ').title() for q in question_types], rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            safe_metric_name = metric_name.replace(' ', '_').replace('[', '').replace(']', '')
            plot_path = os.path.join(self.output_dir, f"{safe_metric_name}_by_question_type.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"      ✅ Saved: {safe_metric_name}_by_question_type.png")
    
    def generate_summary_statistics(self, llm_df: pd.DataFrame, reviewer_df: pd.DataFrame):
        """Generate summary statistics for the analysis."""
        print(f"\n📋 Generating summary statistics...")
        
        # Combine data
        combined_df = pd.concat([llm_df, reviewer_df], ignore_index=True)
        
        # Create summary
        summary = {
            "total_entries": len(combined_df),
            "llm_entries": len(llm_df),
            "reviewer_entries": len(reviewer_df),
            "metrics_analyzed": combined_df['metric_name'].unique().tolist(),
            "question_types": combined_df['question_type'].unique().tolist(),
            "data_types": combined_df['data_type'].unique().tolist()
        }
        
        # Add metric-specific statistics
        metric_stats = {}
        for metric_name in combined_df['metric_name'].unique():
            metric_data = combined_df[combined_df['metric_name'] == metric_name]
            
            if metric_name == "Correctness":
                metric_stats[metric_name] = {
                    "total_evaluations": len(metric_data),
                    "success_rate": metric_data['success'].mean(),
                    "success_count": metric_data['success'].sum(),
                    "failure_count": (~metric_data['success']).sum()
                }
            else:
                metric_stats[metric_name] = {
                    "total_evaluations": len(metric_data),
                    "mean_score": metric_data['score'].mean(),
                    "std_score": metric_data['score'].std(),
                    "min_score": metric_data['score'].min(),
                    "max_score": metric_data['score'].max()
                }
        
        summary["metric_statistics"] = metric_stats
        
        # Save summary
        summary_path = os.path.join(self.output_dir, "analysis_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✅ Summary saved: analysis_summary.json")
        
        # Print key statistics
        print(f"\n📊 Summary Statistics:")
        print(f"  Total entries: {summary['total_entries']}")
        print(f"  LLM entries: {summary['llm_entries']}")
        print(f"  Reviewer entries: {summary['reviewer_entries']}")
        print(f"  Metrics analyzed: {len(summary['metrics_analyzed'])}")
        print(f"  Question types: {len(summary['question_types'])}")
        
        return summary
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("🚀 Starting DeepEval Results Analysis...")
        
        try:
            # Load results files
            llm_results, reviewer_results = self.load_results_files()
            
            # Clean and merge results
            cleaned_llm = self.clean_and_merge_results(llm_results, "llm_vs_reviewer")
            cleaned_reviewer = self.clean_and_merge_results(reviewer_results, "reviewer_vs_reviewer")
            
            # Save merged data
            self.save_merged_data(cleaned_llm, cleaned_reviewer)
            
            # Extract metric data
            llm_df = self.extract_metric_data(cleaned_llm, "llm_vs_reviewer")
            reviewer_df = self.extract_metric_data(cleaned_reviewer, "reviewer_vs_reviewer")
            
            # Create plots
            self.create_comparison_plots(llm_df, reviewer_df)
            
            # Generate summary
            summary = self.generate_summary_statistics(llm_df, reviewer_df)
            
            print(f"\n🎉 Analysis completed successfully!")
            print(f"📁 Output directory: {self.output_dir}")
            print(f"📁 Merged data: {self.merged_data_dir}")
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            raise


def main():
    """Main function to run the analysis."""
    try:
        analyzer = DeepEvalResultsAnalyzer()
        analyzer.run_analysis()
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
