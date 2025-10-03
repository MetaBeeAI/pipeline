#!/usr/bin/env python3
"""
Comprehensive test script for the MetaBeeAI LLM pipeline.
Tests all questions across multiple papers to ensure:
1. Pipeline finds merged_v2.json files in METABEEAI_DATA_DIR/papers/{paper_id}/pages/
2. Output is saved as answers.json in METABEEAI_DATA_DIR/papers/{paper_id}/
3. All questions are processed correctly
4. Output structure contains required fields
"""

import json
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from llm_pipeline import get_answer

# Add parent directory to path for config import
sys.path.append('..')

def get_papers_dir():
    """Get the papers directory from config or use fallback."""
    try:
        from config import get_papers_dir as config_get_papers_dir
        return config_get_papers_dir()
    except ImportError:
        # Fallback to common path
        return "/Users/user/Documents/MetaBeeAI_dataset2/papers"

def get_all_questions() -> Dict[str, str]:
    """Get all questions from the questions.yml file."""
    return {
        "bee_species": "What species of bee(s) were tested?",
        "pesticides": "What pesticide(s) were tested in this study? For each, provide the dose(s), exposure method(s) and duration of exposure.",
        "additional_stressors": "Were the effects of any additional stressors included in the study (like temperature, parasites or pathogens, other chemicals, or diet and nutrition stress)?",
        "experimental_methodology": "What experimental methodologies were used in this paper? Include sample sizes, experimental design, and the level of biological organization measured (e.g., molecular, subindividual, individual, population, or commmunity)",
        "significance": "What are the major findings of the study regarding the effects of the pesticides?",
        "future_research": "Describe any future research directions suggested by the authors in the discussion."
    }

def get_paper_ids(papers_dir: str, count: int = 5) -> List[str]:
    """Get the first N paper IDs from the papers directory."""
    papers_path = Path(papers_dir)
    if not papers_path.exists():
        raise FileNotFoundError(f"Papers directory not found: {papers_dir}")
    
    # Get numeric paper folders
    paper_folders = []
    for folder in papers_path.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            paper_folders.append(folder.name)
    
    # Sort numerically and take first N
    paper_folders.sort(key=int)
    return paper_folders[:count]

def verify_file_structure(papers_dir: str, paper_ids: List[str]) -> Dict[str, bool]:
    """Verify that merged_v2.json files exist in the expected locations."""
    print("🔍 Verifying file structure...")
    structure_status = {}
    
    for paper_id in paper_ids:
        merged_v2_path = Path(papers_dir) / paper_id / "pages" / "merged_v2.json"
        answers_path = Path(papers_dir) / paper_id / "answers.json"
        
        merged_v2_exists = merged_v2_path.exists()
        answers_exists = answers_path.exists()
        
        structure_status[paper_id] = {
            "merged_v2_exists": merged_v2_exists,
            "merged_v2_path": str(merged_v2_path),
            "answers_exists": answers_exists,
            "answers_path": str(answers_path)
        }
        
        status_icon = "✅" if merged_v2_exists else "❌"
        print(f"  {status_icon} Paper {paper_id}: merged_v2.json {'found' if merged_v2_exists else 'NOT FOUND'}")
    
    return structure_status

async def process_paper(paper_id: str, questions: Dict[str, str], papers_dir: str) -> Dict[str, Any]:
    """Process a single paper with all questions."""
    print(f"\n📄 Processing Paper {paper_id}...")
    
    # Path to the merged_v2.json file
    json_path = Path(papers_dir) / paper_id / "pages" / "merged_v2.json"
    
    if not json_path.exists():
        return {
            "paper_id": paper_id,
            "status": "error",
            "error": f"merged_v2.json not found at {json_path}"
        }
    
    results = {}
    
    for question_type, question_text in questions.items():
        print(f"  🤔 Question: {question_type}")
        try:
            # Get answer using the pipeline
            # Note: get_answer is already async, so we can await it directly
            answer_result = await get_answer(question_text, str(json_path))
            
            # Extract the core fields
            if isinstance(answer_result, dict):
                core_result = {
                    "answer": answer_result.get("answer", "No answer generated"),
                    "reason": answer_result.get("reason", "No reason provided"),
                    "chunk_ids": answer_result.get("chunk_ids", [])
                }
                
                # Add metadata if available
                if "relevance_info" in answer_result:
                    core_result["relevance_info"] = answer_result["relevance_info"]
                if "question_metadata" in answer_result:
                    core_result["question_metadata"] = answer_result["question_metadata"]
                if "quality_assessment" in answer_result:
                    core_result["quality_assessment"] = answer_result["quality_assessment"]
                
                results[question_type] = core_result
                print(f"    ✅ Answer generated ({len(core_result.get('chunk_ids', []))} chunks)")
            else:
                results[question_type] = {
                    "answer": str(answer_result),
                    "reason": "Direct response from pipeline",
                    "chunk_ids": []
                }
                print(f"    ✅ Direct response generated")
                
        except Exception as e:
            error_msg = f"Error processing question {question_type}: {str(e)}"
            print(f"    ❌ {error_msg}")
            results[question_type] = {
                "answer": "Error occurred during processing",
                "reason": error_msg,
                "chunk_ids": []
            }
    
    return {
        "paper_id": paper_id,
        "status": "completed",
        "questions": results,
        "timestamp": asyncio.get_event_loop().time()
    }

def save_results(paper_id: str, results: Dict[str, Any], papers_dir: str):
    """Save results to answers.json in the paper folder."""
    answers_path = Path(papers_dir) / paper_id / "answers.json"
    
    # Create the output structure
    output = {
        "QUESTIONS": results["questions"],
        "metadata": {
            "paper_id": paper_id,
            "processing_timestamp": results["timestamp"],
            "pipeline_version": "enhanced_multistage_qa",
            "total_questions": len(results["questions"])
        }
    }
    
    try:
        with open(answers_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"    💾 Results saved to: {answers_path}")
        return True
    except Exception as e:
        print(f"    ❌ Error saving results: {e}")
        return False

async def run_comprehensive_test():
    """Run the comprehensive test across multiple papers."""
    print("🚀 MetaBeeAI LLM Pipeline - Comprehensive Test")
    print("=" * 60)
    
    # Get configuration
    papers_dir = get_papers_dir()
    print(f"📁 Papers directory: {papers_dir}")
    
    # Get questions and paper IDs
    questions = get_all_questions()
    paper_ids = get_paper_ids(papers_dir, count=5)
    
    print(f"📚 Questions to test: {len(questions)}")
    for q_type, q_text in questions.items():
        print(f"  • {q_type}: {q_text[:60]}...")
    
    print(f"\n📄 Papers to test: {len(paper_ids)}")
    for paper_id in paper_ids:
        print(f"  • Paper {paper_id}")
    
    # Verify file structure
    structure_status = verify_file_structure(papers_dir, paper_ids)
    
    # Check if all required files exist
    missing_files = [pid for pid, status in structure_status.items() 
                    if not status["merged_v2_exists"]]
    
    if missing_files:
        print(f"\n❌ Missing merged_v2.json files for papers: {missing_files}")
        print("Please ensure all papers have been processed and deduplicated.")
        return False
    
    print(f"\n✅ All required files found. Starting processing...")
    
    # Process each paper
    all_results = {}
    successful_papers = 0
    
    for paper_id in paper_ids:
        try:
            results = await process_paper(paper_id, questions, papers_dir)
            all_results[paper_id] = results
            
            if results["status"] == "completed":
                successful_papers += 1
                # Save results to answers.json
                save_results(paper_id, results, papers_dir)
            else:
                print(f"  ❌ Paper {paper_id} failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"  ❌ Unexpected error processing paper {paper_id}: {e}")
            all_results[paper_id] = {
                "paper_id": paper_id,
                "status": "error",
                "error": str(e)
            }
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    print(f"Total papers: {len(paper_ids)}")
    print(f"Successful: {successful_papers}")
    print(f"Failed: {len(paper_ids) - successful_papers}")
    
    # Check output structure for successful papers
    print(f"\n🔍 Output Structure Verification:")
    for paper_id, results in all_results.items():
        if results["status"] == "completed":
            questions_results = results["questions"]
            missing_fields = []
            
            for q_type, q_result in questions_results.items():
                required_fields = ["answer", "reason", "chunk_ids"]
                for field in required_fields:
                    if field not in q_result:
                        missing_fields.append(f"{q_type}.{field}")
            
            if missing_fields:
                print(f"  ❌ Paper {paper_id}: Missing fields: {missing_fields}")
            else:
                print(f"  ✅ Paper {paper_id}: All required fields present")
        else:
            print(f"  ❌ Paper {paper_id}: {results.get('error', 'Processing failed')}")
    
    # Save overall test results
    test_summary_path = Path("test_comprehensive_results.json")
    with open(test_summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Test summary saved to: {test_summary_path}")
    
    return successful_papers == len(paper_ids)

if __name__ == "__main__":
    print("Starting comprehensive pipeline test...")
    success = asyncio.run(run_comprehensive_test())
    
    if success:
        print("\n🎉 All tests passed! Pipeline is working correctly.")
    else:
        print("\n💥 Some tests failed. Check the output above for details.")
