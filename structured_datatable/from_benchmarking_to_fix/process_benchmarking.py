#!/usr/bin/env python3
"""
Benchmarking Data Processing Script

This script processes the merged benchmarking data to extract structured information
for comparison between LLM answers and human reviewer answers.

Usage:
    python process_benchmarking.py --question bee_species
    python process_benchmarking.py --question pesticides --interactive
"""

import os
import json
import yaml
import pandas as pd
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
try:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    LLM_AVAILABLE = True
    print("✓ OpenAI client initialized successfully")
except Exception as e:
    print(f"Warning: OpenAI client not available: {e}")
    LLM_AVAILABLE = False

# Try to import structured_datatable functions, with fallbacks
try:
    sys.path.append('../structured_datatable')
    from process_llm_output import (
        load_schema_config, 
        load_field_examples, 
        get_field_examples,
        extract_with_wildcards,
        extract_structured_data
    )
    STRUCTURED_IMPORTS_AVAILABLE = True
    print("✓ Successfully imported structured_datatable functions")
except ImportError as e:
    print(f"Warning: structured_datatable functions not available: {e}")
    print("Using direct LLM extraction instead.")
    STRUCTURED_IMPORTS_AVAILABLE = False
    
    # Define fallback functions
    def load_schema_config(config_file):
        return None
    
    def load_field_examples(examples_file):
        return None
    
    def get_field_examples(field_name, field_type, examples_config):
        return None
    
    def extract_with_wildcards(data, paths):
        return ""
    
    def extract_structured_data(data, schema_config, field_examples):
        return None

class BenchmarkingDataProcessor:
    """Process benchmarking data to extract structured information for comparison."""
    
    def __init__(self, data_dir: str = "llm_benchmarking/final_merged_data", 
                 schema_dir: str = "llm_benchmarking"):
        """
        Initialize the processor.
        
        Args:
            data_dir: Directory containing merged JSON files
            schema_dir: Directory containing schema configuration files
        """
        self.data_dir = Path(data_dir)
        self.schema_dir = Path(schema_dir)
        self.schema_config = None
        self.field_examples = None
        self.available_questions = self._get_available_questions()
        
        # Load configuration files
        self._load_configs()
    
    def _get_available_questions(self) -> List[str]:
        """Get list of available question types from the data directory."""
        if not self.data_dir.exists():
            return []
        
        questions = []
        for json_file in self.data_dir.glob("*_merged.json"):
            question_name = json_file.stem.replace("_merged", "")
            questions.append(question_name)
        
        return sorted(questions)
    
    def _load_configs(self):
        """Load schema configuration and field examples."""
        try:
            schema_file = self.schema_dir / "benchmarking_schema.yaml"
            if schema_file.exists():
                self.schema_config = load_schema_config(str(schema_file))
                print(f"✓ Loaded benchmarking schema from {schema_file}")
            else:
                print(f"Warning: Benchmarking schema not found at {schema_file}")
                self.schema_config = self._get_default_schema()
            
            examples_file = self.schema_dir / "benchmarking_field_examples.yaml"
            if examples_file.exists():
                self.field_examples = load_field_examples(str(examples_file))
                print(f"✓ Loaded benchmarking field examples from {examples_file}")
            else:
                print(f"Warning: Benchmarking field examples not found at {examples_file}")
                self.field_examples = None
                
        except Exception as e:
            print(f"Error loading configurations: {e}")
            self.schema_config = self._get_default_schema()
            self.field_examples = None
    
    def _get_default_schema(self) -> Dict[str, Any]:
        """Get a default schema configuration for benchmarking data."""
        return {
            "output_schema": [
                {
                    "field": "extracted_info",
                    "type": "string",
                    "description": "Extracted structured information from the answer text",
                    "csv_include_field": True
                }
            ],
            "output_formats": ["csv"],
            "data_mappings": {
                "extracted_info": {
                    "category": "EXTRACTED_DATA",
                    "question": "Extract the key information from the answer text in a structured format"
                }
            }
        }
    
    def list_available_questions(self):
        """List all available question types."""
        print("Available question types:")
        for i, question in enumerate(self.available_questions, 1):
            print(f"  {i:2d}. {question}")
        
        if not self.available_questions:
            print("  No questions found. Check the data directory path.")
    
    def process_question(self, question_name: str, output_dir: str = "llm_benchmarking/benchmark_data", max_papers: int = None) -> Optional[str]:
        """
        Process a single question and extract structured data.
        
        Args:
            question_name: Name of the question to process
            output_dir: Directory to save output files
            max_papers: Maximum number of papers to process (None = all papers)
            
        Returns:
            Path to the generated CSV file, or None if processing failed
        """
        if question_name not in self.available_questions:
            print(f"Error: Question '{question_name}' not found.")
            print("Available questions:")
            self.list_available_questions()
            return None
        
        # Load the question data
        json_file = self.data_dir / f"{question_name}_merged.json"
        try:
            with open(json_file, 'r') as f:
                question_data = json.load(f)
        except Exception as e:
            print(f"Error loading data for question '{question_name}': {e}")
            return None
        
        print(f"Processing question: {question_name}")
        print(f"Found {len(question_data)} papers")
        
        # Limit papers if specified
        if max_papers and max_papers > 0:
            limited_data = dict(list(question_data.items())[:max_papers])
            print(f"Processing first {max_papers} papers for testing")
            question_data = limited_data
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Process each paper
        processed_data = []
        optimization_stats = {'all_identical': 0, 'partial_optimization': 0, 'no_optimization': 0}
        
        for paper_id, paper_info in question_data.items():
            try:
                paper_result = self._process_paper(paper_id, paper_info, question_name)
                if paper_result:
                    processed_data.append(paper_result)
                    # Track optimization stats
                    llm_ext = paper_result.get('extracted_llm', '')
                    rev1_ext = paper_result.get('extracted_rev1', '')
                    rev2_ext = paper_result.get('extracted_rev2', '')
                    
                    if llm_ext == rev1_ext == rev2_ext:
                        optimization_stats['all_identical'] += 1
                    elif llm_ext == rev1_ext or llm_ext == rev2_ext or rev1_ext == rev2_ext:
                        optimization_stats['partial_optimization'] += 1
                    else:
                        optimization_stats['no_optimization'] += 1
            except Exception as e:
                print(f"Error processing paper {paper_id}: {e}")
                continue
        
        if not processed_data:
            print("No data was successfully processed.")
            return None
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(processed_data)
        
        # Reorder columns for better readability
        column_order = [
            'paper_id', 'question_type', 'llm_answer', 'rev1_answer', 'rev1_rating', 'rev1_reviewer',
            'rev2_answer', 'rev2_rating', 'rev2_reviewer', 'extracted_llm', 'extracted_rev1', 'extracted_rev2'
        ]
        
        # Only include columns that exist in the data
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        # Save to CSV
        csv_file = output_path / f"{question_name}_extracted.csv"
        df.to_csv(csv_file, index=False)
        
        # Save to JSON
        json_file = output_path / f"{question_name}_extracted.json"
        df.to_json(json_file, orient='records', indent=2)
        
        print(f"Processed {len(processed_data)} papers")
        print(f"Optimization stats:")
        print(f"  - Papers with all identical answers (processed once): {optimization_stats['all_identical']}")
        print(f"  - Papers with partial optimization (2 identical, 1 different): {optimization_stats['partial_optimization']}")
        print(f"  - Papers with no optimization (all different): {optimization_stats['no_optimization']}")
        
        # Calculate API calls saved
        total_saved = (optimization_stats['all_identical'] * 2) + optimization_stats['partial_optimization']
        print(f"  - LLM API calls saved: {total_saved}")
        print(f"Output saved to: {csv_file}")
        print(f"Output saved to: {json_file}")
        
        return str(csv_file)
    
    def _process_paper(self, paper_id: str, paper_info: Dict[str, Any], question_type: str) -> Optional[Dict[str, Any]]:
        """
        Process a single paper's data.
        
        Args:
            paper_id: Paper identifier
            paper_info: Paper information dictionary
            question_type: Type of question being processed
            
        Returns:
            Dictionary with processed paper data
        """
        result = {
            'paper_id': paper_id,
            'question_type': question_type,
            'llm_answer': paper_info.get('answer_llm', ''),
            'rev1_answer': paper_info.get('answer_rev1', ''),
            'rev1_rating': paper_info.get('rev1_rating', ''),
            'rev1_reviewer': paper_info.get('rev1', ''),
            'rev2_answer': paper_info.get('answer_rev2', ''),
            'rev2_rating': paper_info.get('rev2_rating', ''),
            'rev2_reviewer': paper_info.get('rev2', '')
        }
        
        # Clean text by removing italics and normalizing
        result['llm_answer'] = self._clean_text(result['llm_answer'])
        result['rev1_answer'] = self._clean_text(result['rev1_answer'])
        result['rev2_answer'] = self._clean_text(result['rev2_answer'])
        
        # Check if reviewers actually exist before handling fallbacks
        rev1_exists = bool(result['rev1_reviewer'] and result['rev1_reviewer'].strip())
        rev2_exists = bool(result['rev2_reviewer'] and result['rev2_reviewer'].strip())
        
        # Check for identical answers to optimize LLM calls
        answers_identical, primary_answer = self._check_answers_identical(
            result['llm_answer'], result['rev1_answer'], result['rev2_answer']
        )
        
        try:
            if answers_identical:
                # Check which answers are actually identical for optimization
                llm_norm = result['llm_answer'].strip().lower() if result['llm_answer'] else ""
                rev1_norm = result['rev1_answer'].strip().lower() if result['rev1_answer'] else ""
                rev2_norm = result['rev2_answer'].strip().lower() if result['rev2_answer'] else ""
                
                if llm_norm == rev1_norm == rev2_norm:
                    # All three are identical - process once
                    print(f"  Paper {paper_id}: All answers identical, processing once...")
                    if primary_answer:
                        extracted_primary = self._extract_structured_info(primary_answer, question_type)
                        result['extracted_llm'] = extracted_primary
                        result['extracted_rev1'] = extracted_primary if rev1_exists else ''
                        result['extracted_rev2'] = extracted_primary if rev2_exists else ''
                    else:
                        result['extracted_llm'] = ''
                        result['extracted_rev1'] = ''
                        result['extracted_rev2'] = ''
                else:
                    # Two answers are identical, one is different - optimize where possible
                    print(f"  Paper {paper_id}: Partial optimization possible...")
                    
                    # Process the primary answer once
                    if primary_answer:
                        extracted_primary = self._extract_structured_info(primary_answer, question_type)
                    else:
                        extracted_primary = ''
                    
                    # Apply to identical answers, process different ones separately
                    if llm_norm == rev1_norm:
                        result['extracted_llm'] = extracted_primary
                        result['extracted_rev1'] = extracted_primary if rev1_exists else ''
                        # Process rev2 separately if it's different and reviewer exists
                        if rev2_exists and result['rev2_answer'] and rev2_norm != llm_norm:
                            result['extracted_rev2'] = self._extract_structured_info(result['rev2_answer'], question_type)
                        else:
                            result['extracted_rev2'] = extracted_primary if rev2_exists else ''
                    elif llm_norm == rev2_norm:
                        result['extracted_llm'] = extracted_primary
                        result['extracted_rev2'] = extracted_primary if rev2_exists else ''
                        # Process rev1 separately if it's different and reviewer exists
                        if rev1_exists and result['rev1_answer'] and rev1_norm != llm_norm:
                            result['extracted_rev1'] = self._extract_structured_info(result['rev1_answer'], question_type)
                        else:
                            result['extracted_rev1'] = extracted_primary if rev1_exists else ''
                    elif rev1_norm == rev2_norm:
                        result['extracted_rev1'] = extracted_primary if rev1_exists else ''
                        result['extracted_rev2'] = extracted_primary if rev2_exists else ''
                        # Process LLM separately if it's different
                        if result['llm_answer'] and llm_norm != rev1_norm:
                            result['extracted_llm'] = self._extract_structured_info(result['llm_answer'], question_type)
                        else:
                            result['extracted_llm'] = extracted_primary
            else:
                # Process each answer separately if they're all different
                # Extract from LLM answer
                if result['llm_answer']:
                    result['extracted_llm'] = self._extract_structured_info(
                        result['llm_answer'], question_type
                    )
                else:
                    result['extracted_llm'] = ''
                
                # Extract from rev1 answer - only if reviewer exists
                if rev1_exists and result['rev1_answer']:
                    result['extracted_rev1'] = self._extract_structured_info(
                        result['rev1_answer'], question_type
                    )
                else:
                    result['extracted_rev1'] = ''
                
                # Extract from rev2 answer - only if reviewer exists
                if rev2_exists and result['rev2_answer']:
                    result['extracted_rev2'] = self._extract_structured_info(
                        result['rev2_answer'], question_type
                    )
                else:
                    result['extracted_rev2'] = ''
                
        except Exception as e:
            print(f"Error extracting structured info for paper {paper_id}: {e}")
            result['extracted_llm'] = ''
            result['extracted_rev1'] = ''
            result['extracted_rev2'] = ''
        
        return result
    
    def _check_answers_identical(self, llm_answer: str, rev1_answer: str, rev2_answer: str) -> tuple[bool, str]:
        """
        Check if answers are identical or if some are missing/empty.
        Returns True if we can optimize by processing only one answer.
        
        Args:
            llm_answer: LLM answer text
            rev1_answer: Reviewer 1 answer text
            rev2_answer: Reviewer 2 answer text
            
        Returns:
            Tuple of (can_optimize, primary_answer_to_process)
        """
        # Normalize answers for comparison (remove extra whitespace, convert to lowercase)
        llm_norm = llm_answer.strip().lower() if llm_answer else ""
        rev1_norm = rev1_answer.strip().lower() if rev1_answer else ""
        rev2_norm = rev2_answer.strip().lower() if rev2_answer else ""
        
        # Check if any answers are missing/empty/null
        llm_empty = not llm_norm or llm_norm in ['null', 'none', '']
        rev1_empty = not rev1_norm or rev1_norm in ['null', 'none', '']
        rev2_empty = not rev2_norm or rev2_norm in ['null', 'none', '']
        
        # If all answers are identical (including empty), return True
        if llm_norm == rev1_norm == rev2_norm:
            return True, llm_answer
        
        # If two answers are identical and one is empty, return True
        if llm_norm == rev1_norm and rev2_empty:
            return True, llm_answer
        if llm_norm == rev2_norm and rev1_empty:
            return True, llm_answer
        if rev1_norm == rev2_norm and llm_empty:
            return True, rev1_answer
        
        # If two answers are identical (even if third is different), we can optimize
        # by processing the identical pair once and the different one separately
        if llm_norm == rev1_norm:
            return True, llm_answer  # Process LLM/rev1 once, rev2 separately
        if llm_norm == rev2_norm:
            return True, llm_answer  # Process LLM/rev2 once, rev1 separately
        if rev1_norm == rev2_norm:
            return True, rev1_answer  # Process rev1/rev2 once, LLM separately
        
        # If all non-empty answers are identical, return True
        non_empty_answers = [ans for ans in [llm_norm, rev1_norm, rev2_norm] if ans]
        if len(set(non_empty_answers)) == 1:
            return True, non_empty_answers[0]
        
        # All answers are different, need to process separately
        return False, ""
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing markdown formatting and normalizing.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove markdown italics
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        # Remove other markdown formatting
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Code blocks
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_structured_info(self, answer_text: str, question_type: str) -> str:
        """
        Extract structured information from answer text.
        
        Args:
            answer_text: The answer text to process
            question_type: The type of question being processed
            
        Returns:
            Structured information as a string
        """
        if not answer_text:
            return ''
        
        try:
            # Try to use the structured_datatable extraction if available
            if STRUCTURED_IMPORTS_AVAILABLE and self.schema_config:
                # Find the field configuration for this question type
                field_config = None
                for field in self.schema_config.get('output_schema', []):
                    if field['field'] == question_type:
                        field_config = field
                        break
                
                if field_config:
                    print(f"  Using structured_datatable LLM extraction for {question_type}")
                    # Extract using the structured_datatable function
                    extracted = extract_structured_data(
                        answer_text, 
                        field_config, 
                        self.field_examples
                    )
                    
                    if extracted:
                        # Convert to readable string format
                        return self._format_extracted_data(extracted)
                else:
                    print(f"  No field config found for {question_type}, trying direct LLM extraction")
            
            # Try direct LLM extraction if available
            if LLM_AVAILABLE:
                print(f"  Using direct LLM extraction for {question_type}")
                return self._direct_llm_extraction(answer_text, question_type)
            
            # Fallback: simple text processing
            print(f"  Using fallback extraction for {question_type}")
            return self._simple_text_extraction(answer_text, question_type)
            
        except Exception as e:
            print(f"  Error in structured extraction: {e}")
            return self._simple_text_extraction(answer_text, question_type)
    
    def _direct_llm_extraction(self, answer_text: str, question_type: str) -> str:
        """
        Extract structured information using direct LLM API calls.
        
        Args:
            answer_text: The answer text to process
            question_type: The type of question being processed
            
        Returns:
            Structured information as a string
        """
        if not LLM_AVAILABLE:
            return ""
        
        try:
            # Create specific prompts for each question type
            if question_type == "bee_species":
                prompt = f"""Extract bee species information from the following text.

INSTRUCTIONS:
1. Extract ONLY what is explicitly stated in the text
2. Do NOT infer, assume, or add information not present
3. Do NOT convert common names to scientific names unless both are given
4. Maintain the exact level of taxonomic detail provided

EXTRACTION RULES:
- If scientific names (Latin names) are given: return them in standard format (genus species subspecies)
- If only common names are given: return ONLY the common names as written
- If both Latin and common names are given for the same species: return the Latin name only
- If abbreviations are used: expand to full names (e.g., "A. mellifera" → "Apis mellifera")
- If no species information is given: return empty array []
- Do not repeat the same species name in different formats

CRITICAL: When the text says "honey bee" (or "honeybee")  without a scientific name, return "honeybee" NOT "Apis mellifera"

Text: {answer_text}

Return a JSON array. Examples:
- Text says "Apis mellifera carnica": ["Apis mellifera carnica"]
- Text says "Honey bees": ["Honey bees"] 
- Text says "Bumblebees": ["Bumblebees"]
- Text says "honey bees and bumblebees": ["honey bees", "bumblebees"]
- Text says nothing about species: []"""

            elif question_type == "pesticides":
                prompt = f"""Extract pesticide names from the following text.

INSTRUCTIONS:
1. Extract ONLY the names of pesticides that are explicitly mentioned in the text
2. Do NOT infer, assume, or add information not present
3. Use the exact pesticide names as written in the text
4. Do not add any other information or details
5. Only provide the pesticide name, not the commercial formulation, if both are given.

Text: {answer_text}

Return a JSON array of pesticide names:
["pesticide_name_1", "pesticide_name_2", "pesticide_name_3"]

If no pesticide information is found, return an empty array []."""

            elif question_type == "additional_stressors":
                prompt = f"""Extract additional stressors (excluding insecticides) from the following text.

INSTRUCTIONS:
1. Extract ONLY what is explicitly stated in the text
2. Do NOT infer, assume, or add information not present
3. Focus on pathogens, parasites, environmental stressors, non-insecticide chemical stressors, and nutritional stress (starvation, restricion).
4. If a stressor is described in combination with an insecticide stressor, just return the non-insecticide stressor.

EXTRACTION RULES:
- Include only stressors that are explicitly mentioned
- Do not add common stressors that might be expected but aren't stated
- Do not include specific values (measurements, amounts), just the stressor description
- Standardize the wording used for any kind of temperature or thermal stress (return "temperature stress")
- Standardize the wording for any kind of starvation, nutrition, or diet stress (return "nutritional stress")
- If only general terms are used, return those general terms
- If no additional stressors were found, return "none"

CRITICAL: Do not add typical stressors or common examples if they are not in the text

Text: {answer_text}

Return a JSON array of stressors, for example: ["Varroa destructor", "temperature stress", "Nosema infection", "nutritional stress"]

If no additional stressors are found, return "none"."""

            else:
                # Generic extraction for other question types
                prompt = f"""Extract the key information from the following text about {question_type}.

INSTRUCTIONS:
1. Extract ONLY what is explicitly stated in the text
2. Do NOT infer, assume, or add information not present
3. Maintain the exact level of detail provided
4. Focus on factual information rather than interpretations

EXTRACTION RULES:
- Include only information that is directly mentioned
- Do not add common knowledge or typical examples unless stated
- Preserve specific details, numbers, and measurements exactly as written
- If only general terms are used, return those general terms

CRITICAL: Do not add information that might be expected but isn't explicitly stated

Text: {answer_text}

Return a concise summary in JSON format. If no relevant information is found, return an empty object {{}}."""

            # Make the API call
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant. Return only valid JSON with no additional text or explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500
            )
            
            # Get the response content
            raw_content = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                result = json.loads(raw_content)
                return self._format_extracted_data(result)
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw content cleaned up
                return raw_content.replace('```json', '').replace('```', '').strip()
                
        except Exception as e:
            print(f"    Error in direct LLM extraction: {e}")
            return ""
    
    def _format_extracted_data(self, extracted_data: Any) -> str:
        """Format extracted data into a readable string."""
        if not extracted_data:
            return ""
        
        # Handle different data types
        if isinstance(extracted_data, list):
            # For lists (like bee_species), just join the items
            return ", ".join(map(str, extracted_data))
        elif isinstance(extracted_data, dict):
            # For dictionaries, format as key-value pairs
            formatted_parts = []
            for field, value in extracted_data.items():
                if isinstance(value, list):
                    formatted_parts.append(f"{field}: {', '.join(map(str, value))}")
                elif isinstance(value, dict):
                    nested_parts = []
                    for k, v in value.items():
                        if isinstance(v, list):
                            nested_parts.append(f"{k}: {', '.join(map(str, v))}")
                        else:
                            nested_parts.append(f"{k}: {v}")
                    formatted_parts.append(f"{field}: {'; '.join(nested_parts)}")
                else:
                    formatted_parts.append(f"{field}: {value}")
            return "; ".join(formatted_parts)
        else:
            # For other types, return as string
            return str(extracted_data)
    
    def _simple_text_extraction(self, text: str, question_type: str) -> str:
        """
        Simple text extraction when structured extraction fails.
        
        Args:
            text: Text to extract information from
            question_type: Type of question being processed
            
        Returns:
            Extracted information as a string
        """
        if not text:
            return ""
        
        # Clean the text
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove markdown formatting
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        
        # For bee species, extract species names and common names
        if question_type == "bee_species":
            # First try to find scientific names
            species_patterns = [
                r'Apis mellifera(?:\s+\w+)?',  # Honey bees with possible subspecies
                r'Bombus\s+\w+',  # Bumblebees
                r'Osmia\s+\w+',   # Mason bees
                r'Melipona\s+\w+', # Stingless bees
                r'Partamona\s+\w+', # Stingless bees
                r'Scaptotrigona\s+\w+'  # Stingless bees
            ]
            
            found_species = []
            for pattern in species_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                found_species.extend(matches)
            
            if found_species:
                return ', '.join(set(found_species))
            
            # If no scientific names, look for common names
            common_patterns = [
                r'honey\s+bees?',  # Honey bees
                r'bumblebees?',    # Bumblebees
                r'mason\s+bees?',  # Mason bees
                r'stingless\s+bees?'  # Stingless bees
            ]
            
            found_common = []
            for pattern in common_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found_common.append(pattern.replace(r'\s+', ' ').replace(r'\?', ''))
            
            if found_common:
                return ', '.join(found_common)
        
        # For pesticides, extract pesticide names
        elif question_type == "pesticides":
            pesticide_patterns = [
                r'neonicotinoids?',
                r'imidacloprid',
                r'clothianidin',
                r'thiamethoxam',
                r'acetamiprid',
                r'thiacloprid',
                r'organophosphates?',
                r'chlorpyrifos',
                r'pyrethroids?',
                r'lambda-cyhalothrin',
                r'cypermethrin',
                r'deltamethrin'
            ]
            
            found_pesticides = []
            for pattern in pesticide_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found_pesticides.append(pattern)
            
            if found_pesticides:
                return f"Pesticides: {', '.join(set(found_pesticides))}"
        
        # For other question types, return key phrases
        else:
            # Extract key phrases (words that start with capital letters)
            key_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            if key_phrases:
                return f"Key terms: {', '.join(key_phrases[:5])}"  # Limit to first 5
        
        # Fallback: return first 100 characters
        return text[:100] + ("..." if len(text) > 100 else "")
    
    def interactive_mode(self):
        """Run in interactive mode to select questions."""
        if not self.available_questions:
            print("No questions available. Check the data directory.")
            return
        
        print("\n" + "="*50)
        print("BENCHMARKING DATA PROCESSOR")
        print("="*50)
        
        self.list_available_questions()
        print(f"\n  {len(self.available_questions) + 1:2d}. Process all questions")
        print(f"  {len(self.available_questions) + 2:2d}. Exit")
        
        try:
            choice = input("\nSelect an option (number): ").strip()
            
            if choice == str(len(self.available_questions) + 2):
                print("Goodbye!")
                return
            elif choice == str(len(self.available_questions) + 1):
                self._process_all_questions()
            elif choice.isdigit() and 1 <= int(choice) <= len(self.available_questions):
                question_idx = int(choice) - 1
                question_name = self.available_questions[question_idx]
                
                # Ask if user wants to limit papers for testing
                test_limit = input(f"\nProcess all papers for '{question_name}' or limit for testing? (Enter number or 'all'): ").strip()
                
                max_papers = None
                if test_limit.lower() != 'all' and test_limit.isdigit():
                    max_papers = int(test_limit)
                    print(f"Processing first {max_papers} papers for testing...")
                
                print(f"\nProcessing question: {question_name}")
                self.process_question(question_name, max_papers=max_papers)
                print(f"\nQuestion '{question_name}' processed successfully!")
            else:
                print("Invalid choice. Please run the script again with a valid option.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
        except Exception as e:
            print(f"Error: {e}")
    
    def _process_all_questions(self):
        """Process all available questions."""
        print("Processing all questions...")
        
        # Ask if user wants to limit papers for testing
        test_limit = input("\nProcess all papers for all questions or limit for testing? (Enter number or 'all'): ").strip()
        
        max_papers = None
        if test_limit.lower() != 'all' and test_limit.isdigit():
            max_papers = int(test_limit)
            print(f"Processing first {max_papers} papers for all questions (testing mode)...")
        
        for question_name in self.available_questions:
            print(f"\n{'='*60}")
            print(f"Processing: {question_name}")
            print(f"{'='*60}")
            
            try:
                self.process_question(question_name, max_papers=max_papers)
                print(f"✓ {question_name} completed successfully")
            except Exception as e:
                print(f"✗ Error processing {question_name}: {e}")
                continue
        
        print("\nAll questions processed!")

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Process benchmarking data to extract structured information"
    )
    
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Specific question to process (e.g., bee_species, pesticides)"
    )
    
    parser.add_argument(
        "--max-papers", "-m",
        type=int,
        help="Maximum number of papers to process (useful for testing)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="llm_benchmarking/final_merged_data",
        help="Directory containing merged JSON files (default: llm_benchmarking/final_merged_data)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="llm_benchmarking/benchmark_data",
        help="Output directory for extracted data (default: llm_benchmarking/benchmark_data)"
    )
    
    parser.add_argument(
        "--schema-dir",
        type=str,
        default="llm_benchmarking",
        help="Directory containing schema configuration files (default: llm_benchmarking)"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BenchmarkingDataProcessor(
        data_dir=args.data_dir,
        schema_dir=args.schema_dir
    )
    
    if args.interactive:
        processor.interactive_mode()
    elif args.question:
        processor.process_question(args.question, args.output_dir, args.max_papers)
    else:
        print("No question specified. Use --question or --interactive")
        print("Available questions:")
        processor.list_available_questions()

if __name__ == "__main__":
    main()
