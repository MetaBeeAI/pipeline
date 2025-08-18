#!/usr/bin/env python3
"""
Clean script to merge LLM answers with reviewer answers from answers_extended.json files.
This script:
1. Looks for answers_extended.json files in nested directories
2. Extracts reviewer answers and question names
3. Finds corresponding LLM answers from either the same folder or data_dir
4. Creates separate JSON files for each question with merged data
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

def extract_question_name(question_path):
    """
    Extract the final question name from a nested path like 'bee_and_pesticides.bee_species'
    Returns the last part (e.g., 'bee_species')
    """
    return question_path.split('.')[-1]

def find_llm_answer(paper_id, reviewer_folder_path, data_dir):
    """
    Search for LLM answer in two possible locations:
    1. answers.json in the same folder as answers_extended.json
    2. answers.json in data_dir/paper_id/ folder
    """
    # Option 1: Look in the same folder as answers_extended.json
    local_answers_path = os.path.join(reviewer_folder_path, 'answers.json')
    if os.path.exists(local_answers_path):
        try:
            with open(local_answers_path, 'r') as f:
                llm_data = json.load(f)
                return llm_data
        except Exception as e:
            print(f"Error reading local answers.json: {e}")
    
    # Option 2: Look in data_dir/paper_id/answers.json
    data_dir_answers_path = os.path.join(data_dir, str(paper_id), 'answers.json')
    if os.path.exists(data_dir_answers_path):
        try:
            with open(data_dir_answers_path, 'r') as f:
                llm_data = json.load(f)
                return llm_data
        except Exception as e:
            print(f"Error reading data_dir answers.json: {e}")
    
    return None

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

def merge_answers(reviewer_database_path, data_dir, output_dir):
    """
    Main function to merge LLM and reviewer answers
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Dictionary to store merged data for each question
    question_data = defaultdict(dict)
    
    # Track statistics
    total_papers = 0
    total_questions = 0
    
    print(f"Scanning reviewer database: {reviewer_database_path}")
    
    # Walk through the reviewer database structure
    for reviewer_folder in os.listdir(reviewer_database_path):
        reviewer_path = os.path.join(reviewer_database_path, reviewer_folder)
        
        if not os.path.isdir(reviewer_path):
            continue
            
        print(f"Processing reviewer: {reviewer_folder}")
        
        # Look for paper folders within reviewer folder
        for paper_folder in os.listdir(reviewer_path):
            paper_path = os.path.join(reviewer_path, paper_folder)
            
            if not os.path.isdir(paper_path) or not paper_folder.isdigit():
                continue
                
            paper_id = paper_folder
            total_papers += 1
            
            # Look for answers_extended.json
            extended_json_path = os.path.join(paper_path, 'answers_extended.json')
            if not os.path.exists(extended_json_path):
                continue
                
            print(f"  Processing paper {paper_id}")
            
            try:
                # Read reviewer answers
                with open(extended_json_path, 'r') as f:
                    reviewer_data = json.load(f)
                
                if 'QUESTIONS' not in reviewer_data:
                    continue
                
                # Get LLM answers
                llm_data = find_llm_answer(paper_id, paper_path, data_dir)
                
                # Process each question
                for question_path, question_info in reviewer_data['QUESTIONS'].items():
                    if 'user_answer_positive' not in question_info:
                        continue
                        
                    reviewer_answer = question_info['user_answer_positive']
                    if reviewer_answer == "":
                        continue
                    
                    # Extract the final question name
                    question_name = extract_question_name(question_path)
                    total_questions += 1
                    
                    # Get LLM answer for this question
                    llm_answer = ""
                    if llm_data:
                        llm_answer = get_llm_answer_for_question(llm_data, question_path)
                    
                    # Get user rating if available
                    user_rating = question_info.get('user_rating', '')
                    
                    # Check if this paper already exists for this question
                    if paper_id not in question_data[question_name]:
                        # First reviewer for this paper
                        paper_entry = {
                            "answer_llm": llm_answer,
                            "answer_rev1": reviewer_answer,
                            "rev1": reviewer_folder,
                            "rev1_rating": user_rating
                        }
                        question_data[question_name][paper_id] = paper_entry
                    else:
                        # Additional reviewer for this paper
                        existing_entry = question_data[question_name][paper_id]
                        
                        # Count how many reviewers we already have
                        reviewer_count = 1
                        while f"answer_rev{reviewer_count}" in existing_entry:
                            reviewer_count += 1
                        
                        # Add the new reviewer
                        existing_entry[f"answer_rev{reviewer_count}"] = reviewer_answer
                        existing_entry[f"rev{reviewer_count}"] = reviewer_folder
                        existing_entry[f"rev{reviewer_count}_rating"] = user_rating
                    
            except Exception as e:
                print(f"Error processing paper {paper_id}: {e}")
                continue
    
    print(f"\nProcessed {total_papers} papers with {total_questions} total question answers")
    print(f"Found {len(question_data)} unique questions")
    
    # Save separate JSON files for each question
    for question_name, papers in question_data.items():
        output_filename = f"{question_name}_merged.json"
        output_filepath = os.path.join(output_dir, output_filename)
        
        with open(output_filepath, 'w') as f:
            json.dump(papers, f, indent=2)
        
        print(f"Saved {output_filename} with {len(papers)} papers")
    
    print(f"\nMerge complete! Check the '{output_dir}' folder for output files.")

def main():
    parser = argparse.ArgumentParser(description='Merge LLM answers with reviewer answers from answers_extended.json files')
    parser.add_argument('--reviewer-db', type=str, required=True,
                       help='Path to reviewer database folder containing reviewer initials subfolders')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to data directory containing paper folders with LLM answers')
    parser.add_argument('--output-dir', type=str, default='final_merged_data',
                       help='Output directory for merged JSON files (default: final_merged_data)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.reviewer_db):
        print(f"Error: Reviewer database path not found: {args.reviewer_db}")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory path not found: {args.data_dir}")
        return
    
    print("Starting data merge process...")
    print(f"Reviewer database: {args.reviewer_db}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    merge_answers(args.reviewer_db, args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()
