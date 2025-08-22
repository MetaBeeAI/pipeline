#!/usr/bin/env python3
"""
Test script to verify the output structure of the enhanced pipeline.
This script tests that the output contains the required fields: answer, reason, and chunk_ids.
"""

import json
import asyncio
from json_multistage_qa import ask_json

async def test_output_structure():
    """Test that the output structure contains the required fields."""
    
    # Test question
    test_question = "What species of bee(s) were tested?"
    test_json_path = "papers/001/pages/main_p01-02.pdf.json"
    
    print(f"Testing question: {test_question}")
    print(f"Using JSON file: {test_json_path}")
    print("-" * 50)
    
    try:
        # Get the answer
        result = await ask_json(test_question, test_json_path)
        
        print("Raw result:")
        print(json.dumps(result, indent=2))
        print("-" * 50)
        
        # Check required fields
        required_fields = ["answer", "reason", "chunk_ids"]
        missing_fields = [field for field in required_fields if field not in result]
        
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return False
        
        print("✅ All required fields present:")
        for field in required_fields:
            value = result[field]
            if field == "chunk_ids":
                print(f"  {field}: {len(value)} chunk IDs")
                if value:
                    print(f"    First few: {value[:3]}")
            else:
                print(f"  {field}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
        
        # Check additional metadata fields
        additional_fields = ["relevance_info", "question_metadata", "quality_assessment"]
        print("\n📊 Additional metadata fields:")
        for field in additional_fields:
            if field in result:
                print(f"  ✅ {field}: Present")
                if field == "relevance_info":
                    ri = result[field]
                    print(f"    - Total chunks processed: {ri.get('total_chunks_processed', 'N/A')}")
                    print(f"    - Relevant chunks found: {ri.get('relevant_chunks_found', 'N/A')}")
                    print(f"    - Question config: {ri.get('question_config', {}).get('description', 'N/A')}")
            else:
                print(f"  ❌ {field}: Missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    print("Testing Enhanced Pipeline Output Structure")
    print("=" * 50)
    
    success = asyncio.run(test_output_structure())
    
    if success:
        print("\n🎉 All tests passed! Output structure is correct.")
    else:
        print("\n💥 Some tests failed. Check the output structure.")
