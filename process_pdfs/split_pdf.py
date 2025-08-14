#!/usr/bin/env python3
import PyPDF2
import os
import sys
import argparse
from pathlib import Path

def split_pdfs(papers_dir=None):
    """
    Split PDFs in the specified directory into overlapping 2-page segments.
    
    Args:
        papers_dir: Directory containing paper subfolders (defaults to config)
    """
    # Import centralized configuration if papers_dir not provided
    if papers_dir is None:
        import sys
        sys.path.append('..')
        from config import get_papers_dir
        papers_dir = get_papers_dir()
    
    # Validate papers directory
    if not os.path.exists(papers_dir):
        print(f"Error: Directory '{papers_dir}' does not exist")
        return
    
    # Get all subfolders in the specified directory
    subfolders = [f for f in os.listdir(papers_dir) if os.path.isdir(os.path.join(papers_dir, f))]
    
    if not subfolders:
        print(f"No subfolders found in '{papers_dir}'")
        return
    
    print(f"Found {len(subfolders)} subfolders to process")
    
    for subfolder in subfolders:
        # Create pages directory if it doesn't exist
        pages_dir = os.path.join(papers_dir, subfolder, "pages")
        os.makedirs(pages_dir, exist_ok=True)
        
        # Construct path to main PDF using subfolder name
        pdf_path = os.path.join(papers_dir, subfolder, f"{subfolder}_main.pdf")
        
        if not os.path.exists(pdf_path):
            print(f"PDF file not found at {pdf_path}, skipping...")
            continue
        
        try:
            # read the PDF
            print(f"Processing {pdf_path}...")
            pdf_reader = PyPDF2.PdfReader(pdf_path)
            total_pages = len(pdf_reader.pages)
            
            # create overlapping 2-page PDFs
            for i in range(total_pages - 1):  # Stop at second-to-last page
                pdf_writer = PyPDF2.PdfWriter()
                # Add current page and next page
                pdf_writer.add_page(pdf_reader.pages[i])
                pdf_writer.add_page(pdf_reader.pages[i + 1])
                
                output_path = os.path.join(pages_dir, f"main_p{i+1:02d}-{i+2:02d}.pdf")
                with open(output_path, "wb") as output_file:
                    pdf_writer.write(output_file)
            
            print(f"Successfully processed {subfolder}_main.pdf ({total_pages} pages, created {total_pages-1} split PDFs)")
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Split PDFs into overlapping 2-page documents')
    parser.add_argument('directory', type=str, help='Directory containing paper subfolders')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function
    split_pdfs(args.directory)
    print("Processing complete!") 