import json
import os
import argparse
import time
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
async def get_answer(question_text, json_path):
    """
    Retrieves the answer for a given question by calling ask_json.
    Returns a dictionary with the required structure: answer, reason, and chunk_ids.
    """
    result = await ask_json_async(question_text, json_path)
    
    # Ensure the result has the required structure
    if isinstance(result, dict):
        # Extract the required fields from the enhanced result
        return {
            "answer": result.get("answer", ""),
            "reason": result.get("reason", ""),
            "chunk_ids": result.get("chunk_ids", [])
        }
    else:
        # Fallback if result is not a dict
        return {
            "answer": str(result) if result else "",
            "reason": "Answer generated from available information",
            "chunk_ids": []
        }


# ------------------------------------------------------------------------------
# Generic Recursive Function to Process a Hierarchical Question Tree
# ------------------------------------------------------------------------------
async def process_question_tree(tree, json_path, context=None):
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
            answer = await get_answer(question_text, json_path)
            # Process conditional branch if available.
            return answer
        else:
            result = {}
            for key, value in tree.items():
                if key == "list":
                    # If the key is "list", return the list as is.
                    question_of_the_list = value["question"].format(**context)
                    endpoint_name = value["endpoint_name"]
                    answer = await get_answer(question_of_the_list, json_path)
                    list_result = await format_to_list_async(question_of_the_list, answer["answer"])
                    list_items = list_result['answer']
                    result[key] = {}
                    for item in list_items:
                        new_context = context.copy()
                        new_context[endpoint_name] = item
                        result[key][item] = await process_question_tree(value['for_each'], json_path, new_context)
                else:
                    result[key] = await process_question_tree(value, json_path, context)
            return result
    elif isinstance(tree, list):
        return [await process_question_tree(item, json_path, context) for item in tree]
    elif isinstance(tree, str):
        # If the tree itself is a string, treat it as a question.
        question_text = tree.format(**context)
        return await get_answer(question_text, json_path)
    else:
        return tree


# ------------------------------------------------------------------------------
# Main Function: Retrieve All Answers Based on the Questions Dictionary
# ------------------------------------------------------------------------------
async def get_literature_answers(json_path):
    """
    Processes the entire hierarchical question tree defined in QUESTIONS and returns
    the collected answers.
    """
    answers = await process_question_tree(QUESTIONS, json_path)
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

async def process_papers(base_dir=None, start_paper=1, end_paper=999, overwrite_merged=False):
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
    
    total_papers = end_paper - start_paper + 1
    completed_papers = 0
    failed_papers = []
    
    # Create progress log file
    log_file = os.path.join(base_dir, "processing_log.txt")
    
    print(f"🚀 Starting pipeline: {total_papers} papers to process")
    print(f"📁 Papers directory: {base_dir}")
    print(f"📝 Progress log: {log_file}")
    print("=" * 60)
    
    for paper_num in range(start_paper, end_paper + 1):
        # Format the paper number with leading zeros
        paper_folder = f"{paper_num:03d}"
        paper_path = os.path.join(base_dir, paper_folder)
        
        # Show overall progress
        remaining = total_papers - completed_papers
        print(f"\n📊 Progress: {completed_papers}/{total_papers} completed, {remaining} remaining")
        print(f"🔄 Processing paper {paper_folder}...")
        
        # Skip if the paper directory doesn't exist
        if not os.path.exists(paper_path):
            print(f"⏭️  Skipping {paper_folder} - directory not found")
            continue
            
        try:
            pages_path = os.path.join(paper_path, "pages/")
            if not os.path.exists(pages_path):
                print(f"⏭️  Skipping {paper_folder} - pages directory not found")
                continue
                
            # Check if merged_v2.json exists
            json_path = os.path.join(pages_path, "merged_v2.json")
            if not os.path.exists(json_path):
                print(f"⏭️  Skipping {paper_folder} - merged_v2.json not found")
                continue
            
            # Process the paper with progress tracking
            print(f"  📖 Processing {len(QUESTIONS)} questions...")
            
            # Temporarily reduce logging verbosity and suppress all output during processing
            import logging
            import sys
            from io import StringIO
            
            # Capture and suppress all output during processing
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            original_log_level = logging.getLogger().level
            
            # Suppress all output
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            logging.getLogger().setLevel(logging.ERROR)
            
            try:
                literature_answers = await get_literature_answers(json_path)
            finally:
                # Restore all output
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                logging.getLogger().setLevel(original_log_level)
            
            # Only create answers.json for the currently processed paper
            answers_path = os.path.join(paper_path, "answers.json")
            with open(answers_path, 'w') as f:
                json.dump(literature_answers, f, indent=2)
            
            completed_papers += 1
            print(f"  ✅ Paper {paper_folder} completed successfully")
            
            # Log completion
            with open(log_file, 'a') as f:
                f.write(f"{paper_folder}: COMPLETED at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        except Exception as e:
            print(f"  ❌ Error processing paper {paper_folder}: {str(e)}")
            failed_papers.append(paper_folder)
            
            # Log failure
            with open(log_file, 'a') as f:
                f.write(f"{paper_folder}: FAILED at {time.strftime('%Y-%m-%d %H:%M:%S')} - {str(e)}\n")
            continue
    
    # Final summary
    print("\n" + "=" * 60)
    print(f"🎉 PIPELINE COMPLETED!")
    print(f"✅ Successfully processed: {completed_papers}/{total_papers} papers")
    if failed_papers:
        print(f"❌ Failed papers: {', '.join(failed_papers)}")
    print(f"📝 Detailed log: {log_file}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process paper folders to extract literature answers")
    parser.add_argument("--dir", type=str, default=None, 
                      help="Base directory containing paper folders (default: auto-detect from config)")
    parser.add_argument("--start", type=int, default=1, 
                      help="First paper number to process (default: 1)")
    parser.add_argument("--end", type=int, default=999, 
                      help="Last paper number to process (default: 999)")
    parser.add_argument("--overwrite", action="store_true", 
                      help="Overwrite existing merged.json files")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main processing function with the specified parameters
    import asyncio
    asyncio.run(process_papers(
        base_dir=args.dir,
        start_paper=args.start,
        end_paper=args.end,
        overwrite_merged=args.overwrite
    )) 