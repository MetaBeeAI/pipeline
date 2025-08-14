import os
import re
import json
import argparse
from collections import defaultdict
from pathlib import Path

# Import centralized configuration
import sys
sys.path.append('..')
from config import get_papers_dir, get_data_dir

def extract_question_name(question_path):
    """
    Extract the question name from a nested path like 'bee_and_pesticides.bee_species'
    Returns the last part (e.g., 'bee_species')
    """
    return question_path.split('.')[-1]

def get_llm_answer(llm_data, question_path):
    """
    Extract the LLM answer for a specific question path from the original answers.json
    Handles nested structures like 'bee_and_pesticides.bee_species'
    """
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
                    return current['answer']
                else:
                    return ""
        return ""
    except:
        return ""

def get_reviewer_answer_from_extended(reviewer_data, question_path):
    """
    Extract the reviewer answer from answers_extended.json format
    """
    try:
        if question_path in reviewer_data['QUESTIONS']:
            question_data = reviewer_data['QUESTIONS'][question_path]
            if 'user_answer_positive' in question_data:
                answer = question_data['user_answer_positive']
                # If answer is blank, return empty string
                if answer == "":
                    return ""
                return answer
        return ""
    except:
        return ""

def get_reviewer_answer_from_database(reviewer_data, question_path):
    """
    Extract the reviewer answer from external reviewer database format
    """
    try:
        if question_path in reviewer_data['QUESTIONS']:
            question_data = reviewer_data['QUESTIONS'][question_path]
            if 'user_answer_positive' in question_data:
                answer = question_data['user_answer_positive']
                # If answer is blank, return empty string
                if answer == "":
                    return ""
                return answer
        return ""
    except:
        return ""

def merge_data_final(use_extended_json=True, reviewer_database_path=None):
    """
    Merge LLM answers with reviewer answers from either local answers_extended.json files
    or an external reviewer database.
    
    Args:
        use_extended_json (bool): If True, look for answers_extended.json in paper folders
        reviewer_database_path (str): Path to external reviewer database if not using extended JSON
    """
    # Get papers directory from centralized config
    papers_dir = get_papers_dir()
    output_dir = os.path.join(get_data_dir(), 'final_merged_data')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Get all question paths and paper IDs
    all_questions = set()
    reviewer_paper_ids = set()
    
    if use_extended_json:
        # Look for answers_extended.json files in paper folders
        print("Using local answers_extended.json files for reviewer answers")
        
        for paper_folder in os.listdir(papers_dir):
            paper_path = os.path.join(papers_dir, paper_folder)
            if os.path.isdir(paper_path) and paper_folder.isdigit():
                extended_json_path = os.path.join(paper_path, 'answers_extended.json')
                if os.path.exists(extended_json_path):
                    try:
                        with open(extended_json_path, 'r') as f:
                            data = json.load(f)
                            if 'QUESTIONS' in data:
                                all_questions.update(data['QUESTIONS'].keys())
                                reviewer_paper_ids.add(paper_folder)
                    except Exception as e:
                        print(f"Error reading {extended_json_path}: {e}")
                        continue
    else:
        # Use external reviewer database
        if not reviewer_database_path or not os.path.exists(reviewer_database_path):
            print(f"Error: Reviewer database path not found: {reviewer_database_path}")
            return
        
        print(f"Using external reviewer database: {reviewer_database_path}")
        
        # Look for reviewer folders (initials) in the database
        for reviewer_folder in os.listdir(reviewer_database_path):
            reviewer_path = os.path.join(reviewer_database_path, reviewer_folder)
            if os.path.isdir(reviewer_path):
                # Look for paper folders within reviewer folder
                for paper_folder in os.listdir(reviewer_path):
                    paper_path = os.path.join(reviewer_path, paper_folder)
                    if os.path.isdir(paper_path) and paper_folder.isdigit():
                        answers_path = os.path.join(paper_path, 'answers.json')
                        if os.path.exists(answers_path):
                            try:
                                with open(answers_path, 'r') as f:
                                    data = json.load(f)
                                    if 'QUESTIONS' in data:
                                        all_questions.update(data['QUESTIONS'].keys())
                                        reviewer_paper_ids.add(paper_folder)
                            except Exception as e:
                                print(f"Error reading {answers_path}: {e}")
                                continue
    
    print(f"Found {len(all_questions)} unique questions across {len(reviewer_paper_ids)} papers")
    
    # Process each question separately
    for question_path in sorted(all_questions):
        question_name = extract_question_name(question_path)
        print(f"\nProcessing question: {question_name}")
        
        combined_data = {}
        
        # Process each paper that has reviewer answers
        for paper_id in sorted(reviewer_paper_ids):
            print(f"  Processing paper {paper_id}")
            
            # Get LLM answer from papers/{paper_id}/answers.json
            llm_filepath = os.path.join(papers_dir, paper_id, 'answers.json')
            llm_answer = ""
            
            if os.path.exists(llm_filepath):
                try:
                    with open(llm_filepath, 'r') as f:
                        llm_data = json.load(f)
                        llm_answer = get_llm_answer(llm_data, question_path)
                except Exception as e:
                    print(f"Error reading LLM answers for paper {paper_id}: {e}")
                    llm_answer = ""
            
            # Get reviewer answers
            reviewer_answers = {}
            reviewer_initials = {}
            rev_count = 1
            
            if use_extended_json:
                # Get from local answers_extended.json
                extended_json_path = os.path.join(papers_dir, paper_id, 'answers_extended.json')
                if os.path.exists(extended_json_path):
                    try:
                        with open(extended_json_path, 'r') as f:
                            data = json.load(f)
                            reviewer_answer = get_reviewer_answer_from_extended(data, question_path)
                            
                            # If reviewer answer is blank, use LLM answer
                            if reviewer_answer == "":
                                reviewer_answer = llm_answer
                            
                            reviewer_answers[f"answer_rev{rev_count}"] = reviewer_answer
                            reviewer_initials[f"rev{rev_count}"] = "NA"  # Set to "NA" for extended JSON
                            rev_count += 1
                    except Exception as e:
                        print(f"Error reading extended JSON for paper {paper_id}: {e}")
                        continue
            else:
                # Get from external reviewer database
                for reviewer_folder in os.listdir(reviewer_database_path):
                    reviewer_path = os.path.join(reviewer_database_path, reviewer_folder)
                    if os.path.isdir(reviewer_path):
                        paper_path = os.path.join(reviewer_path, paper_id)
                        if os.path.isdir(paper_path):
                            answers_path = os.path.join(paper_path, 'answers.json')
                            if os.path.exists(answers_path):
                                try:
                                    with open(answers_path, 'r') as f:
                                        data = json.load(f)
                                        reviewer_answer = get_reviewer_answer_from_database(data, question_path)
                                        
                                        # If reviewer answer is blank, use LLM answer
                                        if reviewer_answer == "":
                                            reviewer_answer = llm_answer
                                        
                                        reviewer_answers[f"answer_rev{rev_count}"] = reviewer_answer
                                        reviewer_initials[f"rev{rev_count}"] = reviewer_folder
                                        rev_count += 1
                                except Exception as e:
                                    print(f"Error reading reviewer answers for paper {paper_id}, reviewer {reviewer_folder}: {e}")
                                    continue
            
            # Create combined entry for this paper
            combined_data[paper_id] = {
                "answer_llm": llm_answer,
                **reviewer_answers,
                **reviewer_initials
            }
        
        # Save combined data for this question
        output_filename = f"{question_name}_merged.json"
        output_filepath = os.path.join(output_dir, output_filename)
        
        with open(output_filepath, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        print(f"  Saved {output_filename} with {len(combined_data)} papers")

def main():
    parser = argparse.ArgumentParser(description='Merge LLM answers with reviewer answers')
    parser.add_argument('--use-extended', action='store_true', default=True,
                       help='Use local answers_extended.json files (default: True)')
    parser.add_argument('--reviewer-db', type=str, default=None,
                       help='Path to external reviewer database (overrides --use-extended)')
    
    args = parser.parse_args()
    
    # If reviewer database path is specified, use external database
    if args.reviewer_db:
        args.use_extended = False
    
    print("Starting data merge process...")
    if args.use_extended:
        print("Mode: Using local answers_extended.json files")
        print("Reviewer names will be set to 'NA'")
    else:
        print(f"Mode: Using external reviewer database: {args.reviewer_db}")
        print("Reviewer names will be extracted from folder names")
    
    print("\nThis script will:")
    print("1. Read LLM answers from papers/{paper_id}/answers.json")
    if args.use_extended:
        print("2. Read reviewer answers from papers/{paper_id}/answers_extended.json")
    else:
        print(f"2. Read reviewer answers from external database: {args.reviewer_db}")
    print("3. Create merged JSONs for each question")
    print("4. Save output to final_merged_data/ folder")
    
    merge_data_final(use_extended_json=args.use_extended, reviewer_database_path=args.reviewer_db)
    
    print("\nMerge complete! Check the 'final_merged_data' folder for output files.")
    print("Each file contains the combined LLM and reviewer answers for a specific question.")

if __name__ == "__main__":
    main()