import json
import os
import argparse
from json_multistage_qa import ask_json as ask_json_async
from json_multistage_qa import format_to_list as format_to_list_async
import asyncio
import yaml

def ask_json(question_text, json_path):
    """
    Asks a question to the JSON file at the specified path and returns the answer.
    """
    return asyncio.run(ask_json_async(question_text, json_path))


def format_to_list(question,text,model='gpt-4o-mini'):
    """
    Formats the JSON file at the specified path to a list.
    """
    return asyncio.run(format_to_list_async(question, text, model))

# ------------------------------------------------------------------------------
# Hierarchical Questions Dictionary
# ------------------------------------------------------------------------------
# Use {placeholder} format syntax in any question that should be parameterized.

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
questions_path = os.path.join(script_dir, 'questions.yml')

with open(questions_path, 'r') as file:
    QUESTIONS = yaml.safe_load(file)

# ------------------------------------------------------------------------------
# Helper Function: get_answer
# ------------------------------------------------------------------------------
def get_answer(question_text, json_path):
    """
    Retrieves the answer for a given question by calling ask_json.
    Since ask_json returns a dictionary with keys 'reason' and 'answer',
    this function simply extracts the 'answer' field.
    """
    result = ask_json(question_text, json_path)
    return result


# ------------------------------------------------------------------------------
# Generic Recursive Function to Process a Hierarchical Question Tree
# ------------------------------------------------------------------------------
def process_question_tree(tree, json_path, context=None):
    """
    Recursively traverses the question tree (a nested dictionary) and obtains answers using get_answer.

    - If a node contains a "question" key, it is treated as a leaf node.
    - The "for_each" key indicates that the associated value should be processed for
      each item in a list provided via the context.
    - The context is used to format questions with placeholders.
    """
    if context is None:
        context = {}

    # If the tree is a dictionary
    if isinstance(tree, dict):
        # If this dictionary has a "question" key, treat it as a leaf.
        if "question" in tree:
            question_text = tree["question"].format(**context)
            answer = get_answer(question_text, json_path)
            # Process conditional branch if available.
            return answer
        else:
            result = {}
            for key, value in tree.items():
                if key == "list":
                    # If the key is "list", return the list as is.
                    question_of_the_list = value["question"].format(**context)
                    endpoint_name = value["endpoint_name"]
                    answer = get_answer(question_of_the_list, json_path)
                    list_result = format_to_list(question_of_the_list, answer["answer"])
                    list_items = list_result['answer']
                    result[key] = {}
                    for item in list_items:
                        new_context = context.copy()
                        new_context[endpoint_name] = item
                        result[key][item] = process_question_tree(value['for_each'], json_path, new_context)
                else:
                    result[key] = process_question_tree(value, json_path, context)
            return result
    elif isinstance(tree, list):
        return [process_question_tree(item, json_path, context) for item in tree]
    elif isinstance(tree, str):
        # If the tree itself is a string, treat it as a question.
        question_text = tree.format(**context)
        return get_answer(question_text, json_path)
    else:
        return tree


# ------------------------------------------------------------------------------
# Main Function: Retrieve All Answers Based on the Questions Dictionary
# ------------------------------------------------------------------------------
def get_literature_answers(json_path):
    """
    Processes the entire hierarchical question tree defined in QUESTIONS and returns
    the collected answers.
    """
    answers = process_question_tree(QUESTIONS, json_path)
    return answers


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def merge_json_in_the_folder(folder_path, overwrite=False):
    """
    Merges all JSON files in the specified folder into a single dictionary.
    """

    if not overwrite:
        if os.path.exists(folder_path + "merged.json"):
            print("The file already exists. Set 'overwrite=True' to overwrite.")
            return

    chunks_kept = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            json_path = os.path.join(folder_path, file)

            with open(json_path, "r") as f:
                json_obj = json.load(f)

            chunks = json_obj['data']['chunks']

            for chunk in chunks:
                if chunk['chunk_type'] in ['figure','marginalia']:
                    continue
                chunks_kept.append(chunk)

    json_obj = {'data': {
        'chunks':chunks_kept
    }}

    with open(folder_path + "merged.json", "w") as f:
        json.dump(json_obj, f, indent=2)

def process_papers(base_dir=None, start_paper=1, end_paper=999, overwrite_merged=False):
    """
    Processes papers in the specified directory range.
    
    Args:
        base_dir: Base directory containing paper folders (defaults to config)
        start_paper: First paper number to process
        end_paper: Last paper number to process
        overwrite_merged: Whether to overwrite existing merged.json files
    """
    # Import centralized configuration if base_dir not provided
    if base_dir is None:
        import sys
        sys.path.append('..')
        from config import get_papers_dir
        base_dir = get_papers_dir()
    
    # Validate base directory
    if not os.path.exists(base_dir):
        print(f"Error: Base directory '{base_dir}' not found")
        return
    
    # Add trailing slash if missing
    if not base_dir.endswith('/'):
        base_dir += '/'
    
    print(f"Processing papers {start_paper} to {end_paper} in {base_dir}")
    
    for paper_num in range(start_paper, end_paper + 1):
        # Format the paper number with leading zeros
        paper_folder = f"{paper_num:03d}"
        paper_path = os.path.join(base_dir, paper_folder)
        
        print(f"Processing paper {paper_folder}...")
        
        # Skip if the paper directory doesn't exist
        if not os.path.exists(paper_path):
            print(f"Skipping {paper_folder} - directory not found")
            continue
            
        try:
            pages_path = os.path.join(paper_path, "pages/")
            if not os.path.exists(pages_path):
                print(f"Skipping {paper_folder} - pages directory not found")
                continue
                
            merge_json_in_the_folder(pages_path, overwrite=overwrite_merged)
            json_path = os.path.join(pages_path, "merged.json")
            
            if not os.path.exists(json_path):
                print(f"Skipping {paper_folder} - merged.json file not created")
                continue
                
            literature_answers = get_literature_answers(json_path)
            answers_path = os.path.join(paper_path, "answers.json")
            with open(answers_path, 'w') as f:
                json.dump(literature_answers, f, indent=2)
            print(f"Successfully processed paper {paper_folder}")
            
        except Exception as e:
            print(f"Error processing paper {paper_folder}: {str(e)}")
            continue

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process paper folders to extract literature answers")
    parser.add_argument("--dir", type=str, default="../data/papers", 
                      help="Base directory containing paper folders (default: data/papers)")
    parser.add_argument("--start", type=int, default=1, 
                      help="First paper number to process (default: 1)")
    parser.add_argument("--end", type=int, default=999, 
                      help="Last paper number to process (default: 999)")
    parser.add_argument("--overwrite", action="store_true", 
                      help="Overwrite existing merged.json files")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main processing function with the specified parameters
    process_papers(
        base_dir=args.dir,
        start_paper=args.start,
        end_paper=args.end,
        overwrite_merged=args.overwrite
    ) 