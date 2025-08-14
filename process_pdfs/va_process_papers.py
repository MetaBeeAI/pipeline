import requests
import os
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv

def process_papers(papers_dir="../data/papers", start_folder=None):
    """
    Process papers in the specified directory using Vision Agentic Document Analysis, starting from an optional folder number.
    
    Args:
        papers_dir: Directory containing paper subfolders
        start_folder: Optional folder number to start processing from
    """
    # Load environment variables
    load_dotenv()
    
    # Get all subfolders in papers_dir
    if not os.path.exists(papers_dir):
        print(f"Error: Directory '{papers_dir}' does not exist")
        return
        
    subfolders = [f for f in os.listdir(papers_dir) if os.path.isdir(os.path.join(papers_dir, f))]
    # Sort numerically
    subfolders.sort(key=lambda x: int(x))
    
    # If start_folder is specified, filter subfolders
    if start_folder:
        start_idx = next((i for i, folder in enumerate(subfolders) 
                         if int(folder) >= int(start_folder)), len(subfolders))
        subfolders = subfolders[start_idx:]

    url = "https://api.va.landing.ai/v1/tools/agentic-document-analysis"
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(papers_dir, f"processing_log_{timestamp}.txt")
    
    def log_message(message):
        """Write message to both console and log file"""
        print(message)
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    log_message(f"Starting processing in directory: {papers_dir}")
    log_message(f"Found {len(subfolders)} folders to process")
    
    # Process each subfolder in numerical order
    for subfolder in subfolders:
        log_message(f"\nProcessing subfolder: {subfolder}")
        pages_path = os.path.join(papers_dir, subfolder, "pages")
        
        # Make sure directory exists
        if not os.path.exists(pages_path):
            log_message(f"Pages directory not found at {pages_path}, skipping...")
            continue

        # Get list of PDF files and sort them
        page_files = sorted([f for f in os.listdir(pages_path) if f.endswith('.pdf')], 
                          key=lambda x: int(x.split('_p')[1].split('-')[0]))

        if not page_files:
            log_message(f"No PDF files found in {pages_path}, skipping...")
            continue
        
        # Process each page
        for page_file in page_files:
            start_time = time.time()
            
            # Check if JSON exists
            json_path = os.path.join(pages_path, f"{page_file}.json")
            if os.path.exists(json_path):
                log_message(f"JSON file already exists for {page_file}, skipping...")
                continue
                
            file_path = os.path.join(pages_path, page_file)
            
            try:
                with open(file_path, "rb") as f:
                    files = {"pdf": f}
                    headers = {
                        "Authorization": f"Basic {os.getenv('LANDING_AI_API_KEY')}"
                    }
                    
                    response = requests.post(url, files=files, headers=headers)
                    response.raise_for_status()
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # Save response
                    with open(json_path, "w") as f:
                        f.write(response.text)
                    
                    log_message(f"Successfully processed {page_file} in {processing_time:.2f} seconds")
                    
            except Exception as e:
                processing_time = time.time() - start_time
                log_message(f"Error processing {page_file} after {processing_time:.2f} seconds: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Process PDF papers from a specific directory and/or starting folder')
    parser.add_argument('--dir', type=str, help='Papers directory (default: data/papers)', default="data/papers")
    parser.add_argument('--start', type=str, help='Starting folder number (e.g., 059)', default=None)
    args = parser.parse_args()
    
    process_papers(args.dir, args.start)

if __name__ == "__main__":
    main() 