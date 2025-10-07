#!/usr/bin/env python3
"""
Main pipeline runner for PDF processing.
This script runs all steps of the PDF processing pipeline in sequence:
1. Split PDFs into overlapping 2-page segments
2. Process each segment through Vision Agentic API
3. Merge JSON outputs into a single file per paper
4. Deduplicate chunks in merged files

Usage:
    python process_all.py --start 1 --end 10
    python process_all.py --dir /path/to/papers --start 1 --end 10
    python process_all.py --skip-split --skip-api  # Only merge and deduplicate
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Import processing modules
from split_pdf import split_pdfs
from va_process_papers import process_papers
from merger import process_all_papers
from batch_deduplicate import batch_deduplicate

def get_papers_dir():
    """Get the papers directory from config or environment."""
    try:
        sys.path.append('..')
        from config import get_papers_dir as config_get_papers_dir
        return config_get_papers_dir()
    except ImportError:
        # Fallback to common path
        return os.getenv('METABEEAI_DATA_DIR', 'data/papers')

def validate_environment():
    """Check that required environment variables are set."""
    load_dotenv()
    
    required_vars = ['LANDING_AI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("ERROR: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these in your .env file (see ../env.example)")
        return False
    
    return True

def get_all_paper_numbers(papers_dir):
    """Get all numeric paper folder numbers in the directory."""
    if not os.path.exists(papers_dir):
        return []
    
    paper_numbers = []
    for folder in os.listdir(papers_dir):
        folder_path = os.path.join(papers_dir, folder)
        if os.path.isdir(folder_path) and folder.isdigit():
            paper_numbers.append(int(folder))
    
    paper_numbers.sort()
    return paper_numbers

def validate_papers_directory(papers_dir, start_paper, end_paper, merge_only=False):
    """Validate that papers directory exists and contains required PDFs."""
    if not os.path.exists(papers_dir):
        print(f"ERROR: Papers directory not found: {papers_dir}")
        return False
    
    # Check for paper folders in range
    found_papers = []
    for paper_num in range(start_paper, end_paper + 1):
        paper_folder = f"{paper_num:03d}"
        paper_path = os.path.join(papers_dir, paper_folder)
        
        if merge_only:
            # For merge-only mode, check for JSON files instead of PDFs
            pages_dir = os.path.join(paper_path, "pages")
            if os.path.exists(pages_dir):
                json_files = [f for f in os.listdir(pages_dir) if f.endswith('.json') and f.startswith('main_')]
                if json_files:
                    found_papers.append(paper_folder)
        else:
            # For full processing, check for PDF files
            pdf_path = os.path.join(paper_path, f"{paper_folder}_main.pdf")
            if os.path.exists(pdf_path):
                found_papers.append(paper_folder)
    
    if not found_papers:
        if merge_only:
            print(f"ERROR: No JSON files found in range {start_paper:03d}-{end_paper:03d}")
            print(f"Expected files like: {papers_dir}/001/pages/main_*.json")
        else:
            print(f"ERROR: No PDF files found in range {start_paper:03d}-{end_paper:03d}")
            print(f"Expected files like: {papers_dir}/001/001_main.pdf")
        return False
    
    print(f"Found {len(found_papers)} papers to process")
    return True

def run_full_pipeline(papers_dir, start_paper, end_paper, skip_split=False, 
                     skip_api=False, skip_merge=False, skip_deduplicate=False,
                     filter_types=None):
    """
    Run the complete PDF processing pipeline.
    
    Args:
        papers_dir: Directory containing paper subfolders
        start_paper: First paper number to process
        end_paper: Last paper number to process
        skip_split: Skip PDF splitting step
        skip_api: Skip API processing step
        skip_merge: Skip JSON merging step
        skip_deduplicate: Skip deduplication step
        filter_types: List of chunk types to filter out during merging
    """
    print("="*60)
    print("MetaBeeAI PDF Processing Pipeline")
    print("="*60)
    print(f"Papers directory: {papers_dir}")
    print(f"Processing range: {start_paper:03d} to {end_paper:03d}")
    print()
    
    # Step 1: Split PDFs
    if not skip_split:
        print("STEP 1/4: Splitting PDFs into overlapping 2-page segments")
        print("-"*60)
        try:
            split_pdfs(papers_dir)
            print("✓ PDF splitting completed\n")
        except Exception as e:
            print(f"✗ Error during PDF splitting: {e}")
            return False
    else:
        print("STEP 1/4: Skipping PDF splitting (--skip-split)")
        print()
    
    # Step 2: Process through Vision API
    if not skip_api:
        print("STEP 2/4: Processing PDFs through Vision Agentic API")
        print("-"*60)
        print("This step may take a while depending on the number of papers...")
        try:
            process_papers(papers_dir, start_folder=f"{start_paper:03d}")
            print("✓ API processing completed\n")
        except Exception as e:
            print(f"✗ Error during API processing: {e}")
            return False
    else:
        print("STEP 2/4: Skipping API processing (--skip-api)")
        print()
    
    # Step 3: Merge JSON files
    if not skip_merge:
        print("STEP 3/4: Merging JSON files into merged_v2.json")
        print("-"*60)
        try:
            # Get base path (parent of papers dir)
            base_path = str(Path(papers_dir).parent)
            process_all_papers(papers_dir, filter_types or [])
            print("✓ JSON merging completed\n")
        except Exception as e:
            print(f"✗ Error during JSON merging: {e}")
            return False
    else:
        print("STEP 3/4: Skipping JSON merging (--skip-merge)")
        print()
    
    # Step 4: Deduplicate chunks
    if not skip_deduplicate:
        print("STEP 4/4: Deduplicating chunks in merged files")
        print("-"*60)
        try:
            summary = batch_deduplicate(
                base_dir=Path(papers_dir),
                dry_run=False,
                start_paper=start_paper,
                end_paper=end_paper
            )
            print(f"✓ Deduplication completed")
            print(f"  - Processed: {summary.get('processed_papers', 0)} papers")
            print(f"  - Duplicates removed: {summary.get('total_duplicates_removed', 0)}")
            print()
        except Exception as e:
            print(f"✗ Error during deduplication: {e}")
            return False
    else:
        print("STEP 4/4: Skipping deduplication (--skip-deduplicate)")
        print()
    
    # Final summary
    print("="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Processed papers {start_paper:03d} to {end_paper:03d}")
    print(f"\nOutput files created:")
    print(f"  - {papers_dir}/XXX/pages/*.json (individual page JSON files)")
    print(f"  - {papers_dir}/XXX/pages/merged_v2.json (merged and deduplicated)")
    print()
    print("Next step: Run the LLM pipeline to extract information from papers")
    print("  cd ../metabeeai_llm")
    print(f"  python llm_pipeline.py --start {start_paper} --end {end_paper}")
    print()
    
    return True

def main():
    """Main entry point for the pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Process PDFs through the complete MetaBeeAI pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all papers in directory (all steps)
  python process_all.py
  
  # Process papers 1-10 (all steps)
  python process_all.py --start 1 --end 10
  
  # Process papers with custom directory
  python process_all.py --dir /path/to/papers --start 1 --end 10
  
  # Only merge and deduplicate (skip expensive API steps)
  python process_all.py --merge-only
  
  # Merge and deduplicate specific papers
  python process_all.py --merge-only --start 1 --end 10
  
  # Process with chunk type filtering (remove marginalia)
  python process_all.py --start 1 --end 10 --filter-chunk-type marginalia
        """
    )
    
    parser.add_argument(
        '--dir',
        type=str,
        default=None,
        help='Directory containing paper subfolders (defaults to config/env)'
    )
    
    parser.add_argument(
        '--start',
        type=int,
        default=None,
        help='First paper number to process (defaults to first paper in directory)'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='Last paper number to process (defaults to last paper in directory)'
    )
    
    parser.add_argument(
        '--merge-only',
        action='store_true',
        help='Only run merge and deduplication steps (skip expensive PDF splitting and API processing)'
    )
    
    parser.add_argument(
        '--skip-split',
        action='store_true',
        help='Skip PDF splitting step'
    )
    
    parser.add_argument(
        '--skip-api',
        action='store_true',
        help='Skip Vision API processing step'
    )
    
    parser.add_argument(
        '--skip-merge',
        action='store_true',
        help='Skip JSON merging step'
    )
    
    parser.add_argument(
        '--skip-deduplicate',
        action='store_true',
        help='Skip deduplication step'
    )
    
    parser.add_argument(
        '--filter-chunk-type',
        nargs='+',
        default=[],
        help='Chunk types to filter out during merging (e.g., marginalia figure)'
    )
    
    args = parser.parse_args()
    
    # Get papers directory
    papers_dir = args.dir if args.dir else get_papers_dir()
    
    # If merge-only is specified, automatically skip split and API steps
    if args.merge_only:
        args.skip_split = True
        args.skip_api = True
        print("Merge-only mode: Skipping PDF splitting and API processing")
        print()
    
    # Determine paper range
    if args.start is None or args.end is None:
        # Get all papers in directory
        all_papers = get_all_paper_numbers(papers_dir)
        
        if not all_papers:
            print(f"ERROR: No paper folders found in {papers_dir}")
            print("Expected folders with numeric names like: 001, 002, 003, etc.")
            sys.exit(1)
        
        start_paper = args.start if args.start is not None else all_papers[0]
        end_paper = args.end if args.end is not None else all_papers[-1]
        
        print(f"Auto-detected paper range: {start_paper:03d} to {end_paper:03d}")
        print()
    else:
        start_paper = args.start
        end_paper = args.end
    
    # Validate input
    if start_paper < 1:
        print("ERROR: --start must be >= 1")
        sys.exit(1)
    
    if end_paper < start_paper:
        print("ERROR: --end must be >= --start")
        sys.exit(1)
    
    # Validate environment (only if we're running the API step)
    if not args.skip_api:
        if not validate_environment():
            sys.exit(1)
    
    # Validate papers directory
    if not validate_papers_directory(papers_dir, start_paper, end_paper, merge_only=args.merge_only):
        sys.exit(1)
    
    # Run the pipeline
    try:
        success = run_full_pipeline(
            papers_dir=papers_dir,
            start_paper=start_paper,
            end_paper=end_paper,
            skip_split=args.skip_split,
            skip_api=args.skip_api,
            skip_merge=args.skip_merge,
            skip_deduplicate=args.skip_deduplicate,
            filter_types=args.filter_chunk_type
        )
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

