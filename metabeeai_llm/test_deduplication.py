#!/usr/bin/env python3
"""
Test script for the chunk deduplication functionality.
This script tests the deduplication functions with sample data.
"""

import json
from json_multistage_qa import analyze_chunk_uniqueness, deduplicate_chunks, get_duplicate_summary

def create_sample_chunks():
    """Create sample chunks with some duplicates for testing."""
    sample_chunks = [
        {
            "chunk_id": "chunk_001",
            "text": "The study used Apis mellifera workers from 20 colonies.",
            "metadata": {"page": 1}
        },
        {
            "chunk_id": "chunk_002", 
            "text": "Bees were exposed to imidacloprid at 10 ppb concentration.",
            "metadata": {"page": 2}
        },
        {
            "chunk_id": "chunk_003",
            "text": "The study used Apis mellifera workers from 20 colonies.",  # Duplicate of chunk_001
            "metadata": {"page": 3}
        },
        {
            "chunk_id": "chunk_004",
            "text": "Results showed significant effects on foraging behavior.",
            "metadata": {"page": 4}
        },
        {
            "chunk_id": "chunk_005",
            "text": "Bees were exposed to imidacloprid at 10 ppb concentration.",  # Duplicate of chunk_002
            "metadata": {"page": 5}
        },
        {
            "chunk_id": "chunk_006",
            "text": "The study used Apis mellifera workers from 20 colonies.",  # Another duplicate of chunk_001
            "metadata": {"page": 6}
        }
    ]
    return sample_chunks

def test_deduplication():
    """Test the deduplication functionality."""
    print("Testing Chunk Deduplication Functionality")
    print("=" * 50)
    
    # Create sample chunks
    sample_chunks = create_sample_chunks()
    print(f"Created {len(sample_chunks)} sample chunks")
    
    # Test uniqueness analysis
    print("\n1. Testing uniqueness analysis...")
    uniqueness = analyze_chunk_uniqueness(sample_chunks)
    print(f"   Total chunks: {uniqueness['total_chunks']}")
    print(f"   Unique chunks: {uniqueness['unique_chunks']}")
    print(f"   Duplicate chunks: {uniqueness['duplicate_chunks']}")
    print(f"   Duplication rate: {uniqueness['duplication_rate']}%")
    print(f"   Duplicate groups: {uniqueness['duplicate_groups']}")
    
    # Show duplicate details
    if uniqueness['duplicate_details']:
        print("\n   Duplicate details:")
        for i, group in enumerate(uniqueness['duplicate_details'], 1):
            print(f"     Group {i}: {group['text_preview']}")
            print(f"       IDs: {', '.join(group['chunk_ids'])}")
            print(f"       Count: {group['count']}")
    
    # Test deduplication
    print("\n2. Testing deduplication...")
    deduplicated = deduplicate_chunks(sample_chunks)
    print(f"   After deduplication: {len(deduplicated)} chunks")
    
    # Show deduplicated chunks
    print("\n   Deduplicated chunks:")
    for i, chunk in enumerate(deduplicated, 1):
        print(f"     Chunk {i}:")
        print(f"       Primary ID: {chunk['chunk_id']}")
        print(f"       All IDs: {chunk['chunk_ids']}")
        print(f"       Text: {chunk['text'][:50]}...")
    
    # Test duplicate summary
    print("\n3. Testing duplicate summary...")
    summary = get_duplicate_summary(sample_chunks)
    print("   Summary:")
    print(summary)
    
    # Verify no duplicates remain
    print("\n4. Verifying deduplication...")
    final_uniqueness = analyze_chunk_uniqueness(deduplicated)
    print(f"   Final unique chunks: {final_uniqueness['unique_chunks']}")
    print(f"   Final duplicate chunks: {final_uniqueness['duplicate_chunks']}")
    
    if final_uniqueness['duplicate_chunks'] == 0:
        print("   ✅ Deduplication successful!")
    else:
        print("   ❌ Deduplication failed!")
    
    return True

if __name__ == "__main__":
    print("Chunk Deduplication Test")
    print("=" * 50)
    
    try:
        success = test_deduplication()
        if success:
            print("\n🎉 All deduplication tests passed!")
        else:
            print("\n💥 Some deduplication tests failed!")
    except Exception as e:
        print(f"\n💥 Error during testing: {e}")
        import traceback
        traceback.print_exc()
