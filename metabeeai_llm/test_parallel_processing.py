#!/usr/bin/env python3
"""
Test script to demonstrate the parallel processing capabilities of the updated pipeline.
This script shows the hybrid model approach: GPT-4o-mini for relevance scoring (fast) and GPT-4o for answer generation (high quality).
"""

import asyncio
import time
import json
from pathlib import Path
from json_multistage_qa import ask_json, RELEVANCE_MODEL, ANSWER_MODEL, DEFAULT_RELEVANCE_BATCH_SIZE, DEFAULT_ANSWER_BATCH_SIZE

async def test_parallel_processing():
    """Test the parallel processing capabilities with timing information."""
    
    print("🚀 Testing Parallel Processing Pipeline")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"  • Relevance Model: {RELEVANCE_MODEL} (Fast)")
    print(f"  • Answer Model: {ANSWER_MODEL} (High Quality)")
    print(f"  • Relevance Batch Size: {DEFAULT_RELEVANCE_BATCH_SIZE}")
    print(f"  • Answer Batch Size: {DEFAULT_ANSWER_BATCH_SIZE}")
    print()
    
    # Test question
    test_question = "What species of bee(s) were tested?"
    
    # Try to get path from config, fallback to hardcoded path
    try:
        import sys
        sys.path.append('..')
        from config import get_papers_dir
        papers_dir = get_papers_dir()
        test_json_path = Path(papers_dir) / "002" / "pages" / "merged_v2.json"
        print(f"📁 Using config path: {test_json_path}")
    except ImportError:
        # Fallback to hardcoded path
        test_json_path = Path("/Users/user/Documents/MetaBeeAI_dataset2/papers/002/pages/merged_v2.json")
        print(f"📁 Using fallback path: {test_json_path}")
    
    if not test_json_path.exists():
        print(f"❌ Test file not found: {test_json_path}")
        return
    
    print(f"🤔 Testing question: {test_question}")
    print(f"📄 Using JSON file: {test_json_path}")
    print("-" * 60)
    
    # Test with timing
    start_time = time.time()
    
    try:
        print("⏳ Starting pipeline processing...")
        result = await ask_json(test_question, str(test_json_path))
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Processing completed in {processing_time:.2f} seconds")
        print()
        
        # Display results
        print("📊 Results Summary:")
        print(f"  • Answer: {result.get('answer', 'No answer')[:100]}...")
        print(f"  • Reason: {result.get('reason', 'No reason')[:100]}...")
        print(f"  • Chunk IDs: {len(result.get('chunk_ids', []))} chunks")
        
        if 'relevance_info' in result:
            ri = result['relevance_info']
            print(f"  • Total chunks processed: {ri.get('total_chunks_processed', 'N/A')}")
            print(f"  • Relevant chunks found: {ri.get('relevant_chunks_found', 'N/A')}")
        
        print()
        print("🔍 Performance Analysis:")
        print(f"  • Total processing time: {processing_time:.2f}s")
        
        # Estimate speedup from parallel processing
        if 'relevance_info' in result:
            total_chunks = result['relevance_info'].get('total_chunks_processed', 0)
            if total_chunks > 0:
                # Rough estimate: sequential would take ~2s per chunk for relevance + ~3s per chunk for answers
                estimated_sequential_time = total_chunks * 2 + len(result.get('chunk_ids', [])) * 3
                speedup = estimated_sequential_time / processing_time if processing_time > 0 else 0
                print(f"  • Estimated sequential time: {estimated_sequential_time:.1f}s")
                print(f"  • Parallel processing speedup: {speedup:.1f}x")
        
        print()
        print("🎯 Hybrid Model Benefits:")
        print(f"  • Fast relevance scoring with {RELEVANCE_MODEL}")
        print(f"  • High-quality answers with {ANSWER_MODEL}")
        print(f"  • Parallel processing reduces total time")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting parallel processing test...")
    asyncio.run(test_parallel_processing())
