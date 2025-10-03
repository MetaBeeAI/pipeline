#!/usr/bin/env python3
"""
Script to append new LLM answers to existing final_merged_data JSON files.
This script:
1. Reads existing merged data files from final_merged_data/
2. Finds new LLM answers from METABEEAI_DATA_DIR
3. Appends new LLM answers as 'answer_llm2' to existing entries
4. Preserves all existing data without overwriting
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

def load_env_variable(env_file_path=None):
    """
    Load METABEEAI_DATA_DIR from environment or .env file
    """
    # First check environment variable
    data_dir = os.environ.get('METABEEAI_DATA_DIR')
    if data_dir:
        return data_dir
    
    # If no env file specified, try to find .env in current directory or parent
    if not env_file_path:
        current_dir = Path(__file__).parent.parent  # Go up to pipeline root
        env_file_path = current_dir / '.env'
    
    # Try to read from .env file
    if os.path.exists(env_file_path):
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('METABEEAI_DATA_DIR='):
                    data_dir = line.split('=', 1)[1].strip()
                    # Remove quotes if present
                    if data_dir.startswith('"') and data_dir.endswith('"'):
                        data_dir = data_dir[1:-1]
                    elif data_dir.startswith("'") and data_dir.endswith("'"):
                        data_dir = data_dir[1:-1]
                    return data_dir
    
    return None

def extract_question_name(question_path):
    """
    Extract the final question name from a nested path like 'bee_and_pesticides.bee_species'
    Returns the last part (e.g., 'bee_species')
    """
    return question_path.split('.')[-1]

def get_llm_answer_for_question(llm_data, question_path):
    """
    Extract the LLM answer for a specific question path
    """
    if not llm_data or 'QUESTIONS' not in llm_data:
        return ""
    
    try:
        # Handle nested paths like 'bee_and_pesticides.bee_species'
        if '.' in question_path:
            parts = question_path.split('.')
            current = llm_data['QUESTIONS']
            for part in parts:
                if part in current:
                    current = current[part]
                else:
                    return ""
            
            # Check if we have an 'answer' field
            if isinstance(current, dict) and 'answer' in current:
                return current['answer']
            else:
                return ""
        else:
            # Direct question like 'experimental_methodology'
            if question_path in llm_data['QUESTIONS']:
                question_data = llm_data['QUESTIONS'][question_path]
                if isinstance(question_data, dict) and 'answer' in question_data:
                    return question_data['answer']
                else:
                    return ""
        return ""
    except:
        return ""

def find_llm_answer_v2(paper_id, new_data_dir):
    """
    Find LLM answer from the new data directory
    """
    papers_dir = os.path.join(new_data_dir, 'papers')
    answers_path = os.path.join(papers_dir, str(paper_id), 'answers.json')
    
    if os.path.exists(answers_path):
        try:
            with open(answers_path, 'r') as f:
                llm_data = json.load(f)
                return llm_data
        except Exception as e:
            print(f"Error reading new LLM answers for paper {paper_id}: {e}")
    
    return None

def append_llm_v2_answers(final_merged_dir, new_data_dir):
    """
    Append new LLM answers as 'answer_llm2' to existing merged data files
    """
    if not os.path.exists(final_merged_dir):
        print(f"Error: Final merged data directory not found: {final_merged_dir}")
        return
    
    if not os.path.exists(new_data_dir):
        print(f"Error: New data directory not found: {new_data_dir}")
        return
    
    papers_dir = os.path.join(new_data_dir, 'papers')
    if not os.path.exists(papers_dir):
        print(f"Error: Papers directory not found: {papers_dir}")
        return
    
    print(f"Processing final merged data directory: {final_merged_dir}")
    print(f"New LLM data directory: {new_data_dir}")
    
    # Get all merged JSON files
    merged_files = [f for f in os.listdir(final_merged_dir) if f.endswith('_merged.json')]
    
    if not merged_files:
        print("No merged JSON files found in final_merged_data directory")
        return
    
    # Track statistics
    total_files_processed = 0
    total_papers_updated = 0
    total_papers_not_found = 0
    
    # Process each merged file
    for merged_file in merged_files:
        merged_file_path = os.path.join(final_merged_dir, merged_file)
        question_name = merged_file.replace('_merged.json', '')
        
        print(f"\nProcessing {merged_file}...")
        
        # Load existing merged data
        try:
            with open(merged_file_path, 'r') as f:
                merged_data = json.load(f)
        except Exception as e:
            print(f"Error reading {merged_file}: {e}")
            continue
        
        papers_updated_in_file = 0
        papers_not_found_in_file = 0
        
        # Process each paper in the merged data
        for paper_id, paper_data in merged_data.items():
            # Find new LLM answer for this paper
            new_llm_data = find_llm_answer_v2(paper_id, new_data_dir)
            
            if new_llm_data:
                # Get the LLM answer for this specific question
                # We need to determine the question path - for now, assume direct mapping
                # This might need adjustment based on your question structure
                question_paths_to_try = [
                    question_name,  # Direct mapping
                    f"bee_and_pesticides.{question_name}",  # Nested under bee_and_pesticides
                    # Add more patterns if needed
                ]
                
                new_llm_answer = ""
                for question_path in question_paths_to_try:
                    new_llm_answer = get_llm_answer_for_question(new_llm_data, question_path)
                    if new_llm_answer:
                        break
                
                # Add the new LLM answer as 'answer_llm2'
                paper_data['answer_llm2'] = new_llm_answer
                papers_updated_in_file += 1
                
                if new_llm_answer:
                    print(f"  Added answer_llm2 for paper {paper_id}")
                else:
                    print(f"  Added empty answer_llm2 for paper {paper_id} (no matching question found)")
            else:
                papers_not_found_in_file += 1
                print(f"  No new LLM data found for paper {paper_id}")
        
        # Save the updated merged data
        try:
            with open(merged_file_path, 'w') as f:
                json.dump(merged_data, f, indent=2)
            print(f"  Updated {merged_file} - {papers_updated_in_file} papers updated, {papers_not_found_in_file} papers not found")
            total_files_processed += 1
            total_papers_updated += papers_updated_in_file
            total_papers_not_found += papers_not_found_in_file
        except Exception as e:
            print(f"Error saving {merged_file}: {e}")
    
    print(f"\n=== Summary ===")
    print(f"Files processed: {total_files_processed}")
    print(f"Total papers updated: {total_papers_updated}")
    print(f"Total papers not found in new data: {total_papers_not_found}")
    print(f"\nMerge complete! Updated files in '{final_merged_dir}'")

def main():
    parser = argparse.ArgumentParser(description='Append new LLM answers to existing final_merged_data files')
    parser.add_argument('--final-merged-dir', type=str, 
                       default='llm_benchmarking/final_merged_data',
                       help='Path to final_merged_data directory (default: llm_benchmarking/final_merged_data)')
    parser.add_argument('--new-data-dir', type=str,
                       help='Path to new data directory containing LLM answers (if not provided, will use METABEEAI_DATA_DIR)')
    parser.add_argument('--env-file', type=str,
                       help='Path to .env file (default: look for .env in project root)')
    
    args = parser.parse_args()
    
    # Determine new data directory
    new_data_dir = args.new_data_dir
    if not new_data_dir:
        new_data_dir = load_env_variable(args.env_file)
        if not new_data_dir:
            print("Error: Could not determine new data directory.")
            print("Either provide --new-data-dir argument or set METABEEAI_DATA_DIR environment variable or .env file")
            return
    
    # Convert relative paths to absolute
    if not os.path.isabs(args.final_merged_dir):
        script_dir = Path(__file__).parent.parent  # Go up to pipeline root
        final_merged_dir = script_dir / args.final_merged_dir
    else:
        final_merged_dir = args.final_merged_dir
    
    # Validate paths
    if not os.path.exists(final_merged_dir):
        print(f"Error: Final merged data directory not found: {final_merged_dir}")
        return
    
    if not os.path.exists(new_data_dir):
        print(f"Error: New data directory not found: {new_data_dir}")
        return
    
    print("Starting LLM v2 merge process...")
    print(f"Final merged data directory: {final_merged_dir}")
    print(f"New data directory: {new_data_dir}")
    
    append_llm_v2_answers(final_merged_dir, new_data_dir)

if __name__ == "__main__":
    main()
