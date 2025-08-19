#!/usr/bin/env python3
"""
Script to compare reviewer answers and create gold answers with scoring.
Compares extracted_rev1 and extracted_rev2 to create extracted_gold and rev_score.
"""

import json
import argparse
import glob
import re
import ast
import os
import datetime
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher


def parse_list_string(text: str) -> List[str]:
    """
    Parse a string that might be a JSON list, comma-separated list, or single item.
    Returns a list of cleaned strings.
    """
    if not text or text.strip() == "":
        return []
    
    # Clean the text first - order matters: Unicode first, then prefixes
    cleaned_text = clean_unicode_characters(text)
    cleaned_text = clean_text_prefixes(cleaned_text)
    
    # Try to parse as JSON first
    try:
        if cleaned_text.strip().startswith('[') and cleaned_text.strip().endswith(']'):
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, list):
                return [clean_text_prefixes(clean_unicode_characters(str(item).strip())) for item in parsed if str(item).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    
    # If not JSON, try comma-separated
    if ',' in cleaned_text:
        items = [clean_text_prefixes(clean_unicode_characters(item.strip())) for item in cleaned_text.split(',') if item.strip()]
        return items
    
    # Single item
    return [cleaned_text] if cleaned_text else []


def is_nested_json_structure(text: str) -> bool:
    """
    Check if the text represents a nested JSON structure (like pesticides data).
    Handles both JSON and Python dictionary strings.
    """
    if not text or text.strip() == "":
        return False
    
    # First try to parse as JSON
    try:
        parsed = json.loads(text)
        # Check if it's a list of dictionaries with compound_name
        if isinstance(parsed, list) and len(parsed) > 0:
            return isinstance(parsed[0], dict) and "compound_name" in parsed[0]
        # Check if it's a single dictionary with compound_name
        elif isinstance(parsed, dict):
            return "compound_name" in parsed
    except (json.JSONDecodeError, TypeError):
        pass
    
    # If JSON parsing fails, try to detect Python dictionary structure
    text_clean = text.strip()
    
    # Check for Python dictionary indicators
    if (text_clean.startswith('{') and text_clean.endswith('}')) or \
       (text_clean.startswith('[') and text_clean.endswith(']')):
        
        # Look for compound_name in the text
        if "'compound_name'" in text_clean or '"compound_name"' in text_clean:
            return True
    
    return False


def parse_nested_json_structure(text: str) -> List[Dict]:
    """
    Parse nested JSON structure and return list of compound dictionaries.
    Handles both JSON and Python dictionary strings with comprehensive cleaning.
    """
    if not text or text.strip() == "":
        return []
    
    # Step 1: Clean the text thoroughly
    cleaned_text = clean_json_text(text)
    
    # Step 2: Try to parse as JSON first
    try:
        parsed = json.loads(cleaned_text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
        else:
            return []
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Step 3: Try to parse as Python dictionary string
    try:
        parsed = ast.literal_eval(cleaned_text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
        elif isinstance(parsed, tuple):
            # Handle tuples (multiple compounds)
            return list(parsed)
        else:
            return []
    except (ValueError, SyntaxError):
        pass
    
    # Step 4: If all else fails, try a very aggressive approach
    try:
        # Handle common Unicode issues and malformed strings
        import re
        
        # Note: Unicode characters are already cleaned by clean_json_text above
        # This section handles additional malformed patterns
        
        # Try to fix common malformed patterns
        # Fix missing quotes around string values
        cleaned_text = re.sub(r'(\w+):\s*([^,\]]+?)(?=,|\]|$)', r'\1: "\2"', cleaned_text)
        
        # Fix missing quotes around keys
        cleaned_text = re.sub(r'(\w+):', r'"\1":', cleaned_text)
        
        # Try to parse the cleaned text
        try:
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
        except:
            pass
            
        # If JSON still fails, try ast.literal_eval on cleaned text
        parsed = ast.literal_eval(cleaned_text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
        elif isinstance(parsed, tuple):
            # Handle tuples (multiple compounds)
            return list(parsed)
        else:
            return []
            
    except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
        pass
    
    return []


def clean_json_text(text: str) -> str:
    """
    Comprehensive cleaning of JSON text to handle formatting issues.
    """
    if not text:
        return text
    
    # Clean Unicode characters and prefixes first - order matters: Unicode first, then prefixes
    text = clean_unicode_characters(text)
    text = clean_text_prefixes(text)
    
    # Remove newlines, tabs, and extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Handle mixed quote styles - convert to consistent single quotes for Python dict format
    # First, check if it's mostly JSON (double quotes) or Python dict (single quotes)
    double_quote_count = text.count('"')
    single_quote_count = text.count("'")
    
    if double_quote_count > single_quote_count:
        # Convert JSON to Python dict format
        text = text.replace('"', "'")
        # Handle Python literals
        text = text.replace('null', 'None')
        text = text.replace('true', 'True')
        text = text.replace('false', 'False')
    else:
        # Ensure consistent single quotes
        text = text.replace('"', "'")
    
    # Remove any remaining formatting artifacts
    text = re.sub(r'\\n', '', text)
    text = re.sub(r'\\t', '', text)
    text = re.sub(r'\\r', '', text)
    
    # Clean up extra spaces around punctuation
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'\s*:\s*', ': ', text)
    text = re.sub(r'\s*\[\s*', '[', text)
    text = re.sub(r'\s*\]\s*', ']', text)
    text = re.sub(r'\s*\{\s*', '{', text)
    text = re.sub(r'\s*\}\s*', '}', text)
    
    return text


def clean_text_prefixes(text: str) -> str:
    """
    Remove unwanted prefixes like "stressor:" from text.
    """
    if not text:
        return text
    
    # Remove common unwanted prefixes
    prefixes_to_remove = [
        "stressor(s):",
        "stressor(s) :",
        "additional stressor(s):",
        "additional stressor(s) :",
        "stressor:",
        "stressor :",
        "additional stressor:",
        "additional stressor :",
        "stressor",
    ]
    
    cleaned_text = text.strip()
    for prefix in prefixes_to_remove:
        if cleaned_text.lower().startswith(prefix.lower()):
            cleaned_text = cleaned_text[len(prefix):].strip()
            break
    
    return cleaned_text


def clean_unicode_characters(text: str) -> str:
    """
    Clean Unicode characters and convert them to readable equivalents.
    """
    if not text:
        return text
    
    # Comprehensive Unicode character replacements
    unicode_replacements = {
        'μ': 'u',  # mu -> u
        'μg': 'ug',  # microgram -> ug
        '·': '.',  # middle dot -> period
        '⁻¹': '^-1',  # superscript minus one -> ^-1
        '⁻²': '^-2',  # superscript minus two -> ^-2
        '⁻³': '^-3',  # superscript minus three -> ^-3
        '⁻⁴': '^-4',  # superscript minus four -> ^-4
        '⁻⁵': '^-5',  # superscript minus five -> ^-5
        '⁻⁶': '^-6',  # superscript minus six -> ^-6
        '⁻⁷': '^-7',  # superscript minus seven -> ^-7
        '⁻⁸': '^-8',  # superscript minus eight -> ^-8
        '⁻⁹': '^-9',  # superscript minus nine -> ^-9
        '⁻⁰': '^-0',  # superscript minus zero -> ^-0
        '²': '^2',    # superscript two -> ^2
        '³': '^3',    # superscript three -> ^3
        '⁴': '^4',    # superscript four -> ^4
        '⁵': '^5',    # superscript five -> ^5
        '⁶': '^6',    # superscript six -> ^6
        '⁷': '^7',    # superscript seven -> ^7
        '⁸': '^8',    # superscript eight -> ^8
        '⁹': '^9',    # superscript nine -> ^9
        '⁰': '^0',    # superscript zero -> ^0
        '₅₀': '50',   # subscript 50 -> 50
        '₃': '3',     # subscript 3 -> 3
        '₄': '4',     # subscript 4 -> 4
        '₅': '5',     # subscript 5 -> 5
        '₆': '6',     # subscript 6 -> 6
        '₇': '7',     # subscript 7 -> 7
        '₈': '8',     # subscript 8 -> 8
        '₉': '9',     # subscript 9 -> 9
        '₀': '0',     # subscript 0 -> 0
        '×': 'x',     # multiplication sign -> x
        '±': '+/-',   # plus-minus -> +/-
        '°': 'deg',   # degree symbol -> deg
        '≤': '<=',    # less than or equal -> <=
        '≥': '>=',    # greater than or equal -> >=
        '≠': '!=',    # not equal -> !=
        '≈': '~',     # approximately equal -> ~
        '∞': 'inf',   # infinity -> inf
        'α': 'alpha', # alpha -> alpha
        'β': 'beta',  # beta -> beta
        'γ': 'gamma', # gamma -> gamma
        'δ': 'delta', # delta -> delta
        'ε': 'epsilon', # epsilon -> epsilon
        'θ': 'theta', # theta -> theta
        'λ': 'lambda', # lambda -> lambda
        'σ': 'sigma', # sigma -> sigma
        'τ': 'tau',   # tau -> tau
        'φ': 'phi',   # phi -> phi
        'χ': 'chi',   # chi -> chi
        'ψ': 'psi',   # psi -> psi
        'ω': 'omega', # omega -> omega
    }
    
    cleaned_text = text
    for unicode_char, replacement in unicode_replacements.items():
        cleaned_text = cleaned_text.replace(unicode_char, replacement)
    
    return cleaned_text


def calculate_similarity_score(item1: str, item2: str) -> float:
    """
    Calculate similarity score between two items.
    0 = completely different, 1 = exactly the same, 0.5 = partial match, 0.9 = abbreviation match
    """
    # First check exact match after stripping non-alphanumeric characters
    stripped_item1 = strip_non_alphanumeric(item1)
    stripped_item2 = strip_non_alphanumeric(item2)
    
    if stripped_item1 == stripped_item2:
        return 1.0
    
    # Check if terms are equivalent after normalization (e.g., "honey bee" vs "honeybee")
    if are_equivalent_terms(item1, item2):
        return 1.0
    
    # Check for abbreviation matches (e.g., "A. mellifera" vs "Apis mellifera")
    if is_abbreviation_match(item1, item2):
        return 0.9
    
    # Check for partial matches (e.g., "A. mellifera" vs "Apis mellifera")
    # Use normalized text for better word comparison
    normalized_item1 = normalize_text(item1)
    normalized_item2 = normalize_text(item2)
    
    item1_words = set(normalized_item1.split())
    item2_words = set(normalized_item2.split())
    
    # Calculate word overlap
    if item1_words and item2_words:
        intersection = item1_words.intersection(item2_words)
        union = item1_words.union(item2_words)
        word_overlap = len(intersection) / len(union) if union else 0
        
        # Use sequence matcher for overall similarity (also on normalized text)
        sequence_similarity = SequenceMatcher(None, normalized_item1, normalized_item2).ratio()
        
        # Combine both metrics
        combined_score = (word_overlap + sequence_similarity) / 2
        
        # Threshold for partial match
        if combined_score > 0.3:
            return 0.5
        else:
            return 0.0
    
    return 0.0


def is_abbreviation_match(item1: str, item2: str) -> bool:
    """
    Check if two items are the same but one uses abbreviations.
    Examples: "A. mellifera" vs "Apis mellifera", "B. terrestris" vs "Bombus terrestris"
    """
    # First check if they're equivalent after stripping non-alphanumeric characters
    stripped_item1 = strip_non_alphanumeric(item1)
    stripped_item2 = strip_non_alphanumeric(item2)
    
    if stripped_item1 == stripped_item2:
        return False  # They're exactly the same, not abbreviation matches
    
    item1_words = item1.lower().split()
    item2_words = item2.lower().split()
    
    # If they have different numbers of words, they can't be abbreviation matches
    if len(item1_words) != len(item2_words):
        return False
    
    # Check each word pair
    for word1, word2 in zip(item1_words, item2_words):
        # If words are the same, continue
        if word1 == word2:
            continue
        
        # Check if one is an abbreviation of the other
        if is_abbreviation(word1, word2) or is_abbreviation(word2, word1):
            continue
        else:
            # If any word pair doesn't match, it's not an abbreviation match
            return False
    
    return True


def is_abbreviation(abbrev: str, full_word: str) -> bool:
    """
    Check if abbrev is an abbreviation of full_word.
    Examples: "a." -> "apis", "b." -> "bombus"
    """
    # Remove punctuation from abbreviation
    abbrev_clean = abbrev.strip('.')
    
    # Check if abbreviation is a single letter followed by punctuation
    if len(abbrev_clean) == 1 and abbrev.endswith('.'):
        # Check if full_word starts with that letter
        if full_word.lower().startswith(abbrev_clean.lower()):
            # Check if it's a reasonable abbreviation (full word should be longer)
            if len(full_word) > 2:
                return True
    
    return False


def normalize_text(text: str) -> str:
    """
    Normalize text by removing taxonomic abbreviations, spaces, and plural forms.
    Examples: "Apis mellifera ssp. carnica" -> "apismelliferacarnica"
    """
    # Clean Unicode characters and prefixes first - order matters: Unicode first, then prefixes
    text = clean_unicode_characters(text)
    text = clean_text_prefixes(text)
    
    # Remove taxonomic abbreviations - be more aggressive
    text = re.sub(r'\b(spp?\.|ssp\.|sp\.)\s*', '', text, flags=re.IGNORECASE)
    
    # Remove extra whitespace and convert to lowercase
    text = re.sub(r'\s+', '', text.lower())
    
    # Remove common plural endings
    text = re.sub(r's$', '', text)  # Remove trailing 's'
    
    return text


def are_equivalent_terms(term1: str, term2: str) -> bool:
    """
    Check if two terms are equivalent after normalization.
    Examples: "honey bee" vs "honeybee" vs "honey bees" -> True
    """
    # First check if they're equivalent after stripping non-alphanumeric characters
    stripped1 = strip_non_alphanumeric(term1)
    stripped2 = strip_non_alphanumeric(term2)
    
    if stripped1 == stripped2:
        return True
    
    # Fall back to the original normalization method
    normalized1 = normalize_text(term1)
    normalized2 = normalize_text(term2)
    
    return normalized1 == normalized2


def is_more_complete(item1: str, item2: str) -> bool:
    """
    Determine if item1 is more complete than item2.
    Returns True if item1 contains all the information from item2 plus more.
    """
    # First check if they're equivalent after stripping non-alphanumeric characters
    stripped_item1 = strip_non_alphanumeric(item1)
    stripped_item2 = strip_non_alphanumeric(item2)
    
    if stripped_item1 == stripped_item2:
        return False  # They're equal, not more complete
    
    # Check if they're equivalent after normalization
    if are_equivalent_terms(item1, item2):
        return False  # They're equivalent, not more complete
    
    # Use stripped versions for word comparison
    item1_words = set(stripped_item1.split())
    item2_words = set(stripped_item2.split())
    
    # Check if item2 is a subset of item1 (item1 contains all words from item2)
    if item2_words.issubset(item1_words):
        # item1 has all words from item2, so it's more complete
        return True
    
    # Check if item1 is a subset of item2 (item2 contains all words from item1)
    if item1_words.issubset(item2_words):
        # item2 has all words from item1, so item1 is NOT more complete
        return False
    
    # If neither is a subset, check if one has significantly more words
    # This handles cases like "Bombus terrestris" vs "Bombus terrestris audax"
    if len(item1_words) > len(item2_words) and len(item1_words) - len(item2_words) >= 1:
        # Check if the additional words in item1 are meaningful additions
        extra_words = item1_words - item2_words
        # If extra words are not just articles, prepositions, etc.
        meaningful_extra = {word for word in extra_words if len(word) > 2}
        if meaningful_extra:
            return True
    
    return False


def compare_answers(rev1_items: List[str], rev2_items: List[str]) -> Tuple[float, List[str]]:
    """
    Compare two lists of answers and return a score and gold answer.
    Uses the same fair scoring logic as LLM comparison.
    
    Args:
        rev1_items: List of items from reviewer 1
        rev2_items: List of items from reviewer 2
    
    Returns:
        Tuple of (overall_score, gold_answer_list)
    """
    if not rev1_items and not rev2_items:
        return 0.0, []
    
    if not rev1_items:
        return 0.0, rev2_items
    
    if not rev2_items:
        # Clean rev1 items before returning them as gold answer
        cleaned_rev1_items = []
        for item in rev1_items:
            cleaned_item = clean_text_prefixes(clean_unicode_characters(item))
            cleaned_rev1_items.append(cleaned_item)
        return 0.0, cleaned_rev1_items
    
    # Create gold answer by intelligently merging both lists
    gold_items = []
    
    # First, add all items from rev1
    for item in rev1_items:
        gold_items.append(item)
    
    # Then process items from rev2
    for rev2_item in rev2_items:
        should_add = True
        item_to_replace = None
        
        # Check if rev2_item is similar to any existing item
        for i, gold_item in enumerate(gold_items):
            similarity = calculate_similarity_score(rev2_item, gold_item)
            
            if similarity >= 0.5:
                # Check if rev2_item is more complete (superset) than gold_item
                if is_more_complete(rev2_item, gold_item):
                    # Replace the less complete item with the more complete one
                    item_to_replace = i
                    should_add = False
                    break
                elif is_more_complete(gold_item, rev2_item):
                    # gold_item is more complete, don't add rev2_item
                    should_add = False
                    break
                else:
                    # They're similar but neither is clearly more complete
                    should_add = False
                    break
        
        if item_to_replace is not None:
            gold_items[item_to_replace] = rev2_item
        elif should_add:
            gold_items.append(rev2_item)
    
    # Calculate agreement score using the same logic as LLM comparison
    # First, deduplicate both reviewer answers by grouping similar items (handles typos)
    unique_rev1_groups = []
    used_rev1_indices = set()
    
    for i, rev1_item in enumerate(rev1_items):
        if i in used_rev1_indices:
            continue
            
        # Start a new group with this item
        current_group = [rev1_item]
        used_rev1_indices.add(i)
        
        # Find similar items to group together
        for j, other_rev1 in enumerate(rev1_items):
            if j in used_rev1_indices:
                continue
            if calculate_similarity_score(rev1_item, other_rev1) >= 0.9:  # Very similar items
                current_group.append(other_rev1)
                used_rev1_indices.add(j)
        
        # Use the most complete item from the group as the representative
        representative = max(current_group, key=len)
        unique_rev1_groups.append(representative)
    
    unique_rev2_groups = []
    used_rev2_indices = set()
    
    for i, rev2_item in enumerate(rev2_items):
        if i in used_rev2_indices:
            continue
            
        # Start a new group with this item
        current_group = [rev2_item]
        used_rev2_indices.add(i)
        
        # Find similar items to group together
        for j, other_rev2 in enumerate(rev2_items):
            if j in used_rev2_indices:
                continue
            if calculate_similarity_score(rev2_item, other_rev2) >= 0.9:  # Very similar items
                current_group.append(other_rev2)
                used_rev2_indices.add(j)
        
        # Use the most complete item from the group as the representative
        representative = max(current_group, key=len)
        unique_rev2_groups.append(representative)
    
    # Count how many unique items both reviewers identified (agreement)
    agreement_count = 0
    rev2_matched_indices = set()
    
    for rev1_representative in unique_rev1_groups:
        best_rev2_match = None
        best_score = 0.0
        
        for i, rev2_representative in enumerate(unique_rev2_groups):
            if i in rev2_matched_indices:
                continue
            score = calculate_similarity_score(rev1_representative, rev2_representative)
            if score > best_score:
                best_score = score
                best_rev2_match = i
        
        # If we found a good match (>= 0.7), count it as agreement
        if best_score >= 0.7 and best_rev2_match is not None:
            agreement_count += 1
            rev2_matched_indices.add(best_rev2_match)
    
    # Calculate agreement score as: agreement_count / total_unique_items
    total_unique_items = len(unique_rev1_groups) + len(unique_rev2_groups) - agreement_count
    overall_score = agreement_count / total_unique_items if total_unique_items > 0 else 0.0
    
    # Clean the final gold items before returning
    cleaned_gold_items = []
    for item in gold_items:
        cleaned_item = clean_text_prefixes(clean_unicode_characters(item))
        cleaned_gold_items.append(cleaned_item)
    
    return overall_score, cleaned_gold_items


def compare_llm_with_gold(llm_items: List[str], gold_items: List[str]) -> Tuple[float, str]:
    """
    Compare LLM answers with gold answers and return similarity score and comparison category.
    
    Returns:
        Tuple of (similarity_score, comparison_category)
        comparison_category: "match", "llm_less", "llm_more", or "conflict"
    """
    if not llm_items and not gold_items:
        return 1.0, "match"  # Both empty is a perfect match
    
    if not llm_items:
        return 0.0, "llm_less"  # LLM has no information
    
    if not gold_items:
        return 0.0, "llm_more"  # LLM has information but gold doesn't
    
    # First, deduplicate gold items by grouping similar items (handles typos)
    unique_gold_groups = []
    used_gold_indices = set()
    
    for i, gold_item in enumerate(gold_items):
        if i in used_gold_indices:
            continue
            
        # Start a new group with this item
        current_group = [gold_item]
        used_gold_indices.add(i)
        
        # Find similar items to group together
        for j, other_gold in enumerate(gold_items):
            if j in used_gold_indices:
                continue
            if calculate_similarity_score(gold_item, other_gold) >= 0.9:  # Very similar items
                current_group.append(other_gold)
                used_gold_indices.add(j)
        
        # Use the most complete item from the group as the representative
        representative = max(current_group, key=len)
        unique_gold_groups.append(representative)
    
    # Now count how many unique gold items the LLM correctly identified
    correct_matches = 0
    llm_matched_indices = set()
    
    for gold_representative in unique_gold_groups:
        best_llm_match = None
        best_score = 0.0
        
        for i, llm_item in enumerate(llm_items):
            if i in llm_matched_indices:
                continue
            score = calculate_similarity_score(llm_item, gold_representative)
            if score > best_score:
                best_score = score
                best_llm_match = i
        
        # If we found a good match (>= 0.7), count it as correct
        if best_score >= 0.7 and best_llm_match is not None:
            correct_matches += 1
            llm_matched_indices.add(best_llm_match)
    
    # Calculate score as: correct_matches / total_unique_gold_items
    overall_score = correct_matches / len(unique_gold_groups) if unique_gold_groups else 0.0
    
    # Determine comparison category
    if overall_score >= 0.9:
        if len(llm_items) == len(unique_gold_groups):
            comparison = "match"
        elif len(llm_items) < len(unique_gold_groups):
            comparison = "llm_less"
        else:
            comparison = "llm_more"
    elif overall_score >= 0.7:
        if len(llm_items) < len(unique_gold_groups):
            comparison = "llm_less"
        elif len(llm_items) > len(unique_gold_groups):
            comparison = "llm_more"
        else:
            comparison = "llm_less"  # Default to less if unclear
    elif overall_score >= 0.3:
        comparison = "llm_less"  # Significant differences
    else:
        comparison = "conflict"  # Completely different
    
    return overall_score, comparison


def compare_nested_json_answers(rev1_compounds: List[Dict], rev2_compounds: List[Dict]) -> Tuple[float, List[Dict]]:
    """
    Compare nested JSON structures while maintaining compound separation.
    
    Args:
        rev1_compounds: List of compound dictionaries from reviewer 1
        rev2_compounds: List of compound dictionaries from reviewer 2
    
    Returns:
        Tuple of (overall_score, gold_compounds_list)
    """
    if not rev1_compounds and not rev2_compounds:
        return 0.0, []
    
    if not rev1_compounds:
        return 0.0, rev2_compounds
    
    if not rev2_compounds:
        return 0.0, rev1_compounds
    
    # Create gold answer by intelligently merging compounds
    # Sort compounds by name for consistent comparison
    def get_compound_name(compound):
        return compound.get('compound_name', '').lower()
    
    rev1_sorted = sorted(rev1_compounds, key=get_compound_name)
    rev2_sorted = sorted(rev2_compounds, key=get_compound_name)
    
    gold_compounds = []
    
    # First, add all compounds from rev1
    for compound in rev1_sorted:
        gold_compounds.append(compound.copy())
    
    # Then process compounds from rev2
    for rev2_compound in rev2_sorted:
        rev2_compound_name = rev2_compound.get("compound_name", "").lower()
        
        # Check if this compound already exists in gold
        existing_compound_idx = None
        for i, gold_compound in enumerate(gold_compounds):
            gold_compound_name = gold_compound.get("compound_name", "").lower()
            
            # Check if compound names match (allowing for abbreviations)
            if are_equivalent_terms(rev2_compound_name, gold_compound_name):
                existing_compound_idx = i
                break
        
        if existing_compound_idx is not None:
            # Merge the existing compound with rev2 data
            gold_compounds[existing_compound_idx] = merge_compound_data(
                gold_compounds[existing_compound_idx], rev2_compound
            )
        else:
            # Add new compound
            gold_compounds.append(rev2_compound.copy())
    
    # Sort final result by compound name for consistency
    gold_compounds.sort(key=get_compound_name)
    
    # Calculate scores for each compound
    total_score = 0.0
    scored_compounds = 0
    
    for rev1_compound in rev1_sorted:
        best_score = 0.0
        for rev2_compound in rev2_sorted:
            score = calculate_compound_similarity(rev1_compound, rev2_compound)
            best_score = max(best_score, score)
        total_score += best_score
        scored_compounds += 1
    
    for rev2_compound in rev2_sorted:
        best_score = 0.0
        for rev1_compound in rev1_sorted:
            score = calculate_compound_similarity(rev2_compound, rev1_compound)
            best_score = max(best_score, score)
        total_score += best_score
        scored_compounds += 1
    
    overall_score = total_score / scored_compounds if scored_compounds > 0 else 0.0
    
    return overall_score, gold_compounds


def calculate_compound_similarity(compound1: Dict, compound2: Dict) -> float:
    """
    Calculate similarity between two compound dictionaries.
    """
    if not compound1 or not compound2:
        return 0.0
    
    # Check compound names first
    name1 = compound1.get("compound_name", "").lower()
    name2 = compound2.get("compound_name", "").lower()
    
    if not are_equivalent_terms(name1, name2):
        return 0.0
    
    # If names match, check other fields
    total_score = 0.0
    field_count = 0
    
    # Compare exposure methods
    methods1 = compound1.get("exposure_methods", [])
    methods2 = compound2.get("exposure_methods", [])
    
    if methods1 and methods2:
        method_score = calculate_list_similarity(methods1, methods2)
        total_score += method_score
        field_count += 1
    
    # Add more field comparisons as needed
    # For now, return a high score if names match
    return 0.8 if field_count > 0 else 0.6


def calculate_list_similarity(list1: List, list2: List) -> float:
    """
    Calculate similarity between two lists of dictionaries.
    """
    if not list1 and not list2:
        return 1.0
    
    if not list1 or not list2:
        return 0.0
    
    # Simple approach: check if they have similar structure
    if len(list1) == len(list2):
        return 0.7
    else:
        return 0.4


def merge_compound_data(compound1: Dict, compound2: Dict) -> Dict:
    """
    Merge two compound dictionaries, keeping the more complete information.
    """
    merged = compound1.copy()
    
    # Merge exposure methods
    methods1 = compound1.get("exposure_methods", [])
    methods2 = compound2.get("exposure_methods", [])
    
    if methods2 and not methods1:
        merged["exposure_methods"] = methods2
    elif methods1 and methods2:
        # Combine methods, avoiding duplicates
        combined_methods = methods1 + methods2
        # Simple deduplication - could be improved
        merged["exposure_methods"] = combined_methods
    
    # Add any fields from compound2 that aren't in compound1
    for key, value in compound2.items():
        if key not in merged or not merged[key]:
            merged[key] = value
    
    return merged


def process_file(file_path: str, method: str = "reviewers") -> None:
    """
    Process a single JSON file to compare reviewer answers and create gold answers,
    or compare LLM with gold answers.
    
    Args:
        file_path: Path to the JSON file to process
        method: Either "reviewers" or "llm"
    """
    print(f"Processing: {file_path} using method: {method}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return
    
    if not isinstance(data, list):
        print(f"File {file_path} does not contain a list of records")
        return
    
    processed_count = 0
    updated_count = 0
    
    # Statistics tracking for reviewer agreement
    perfect_scores = 0
    zero_scores = 0
    total_score = 0.0
    scored_records = 0
    
    # Statistics tracking for LLM comparison (only used in LLM mode)
    llm_scored_records = 0
    llm_total_score = 0.0
    llm_matches = 0
    llm_less = 0
    llm_more = 0
    llm_conflicts = 0
    
    for record in data:
        if not isinstance(record, dict):
            continue
        
        processed_count += 1
        
        if method == "llm":
            # LLM comparison mode - compare LLM with gold
            if "extracted_llm" not in record or not record["extracted_llm"]:
                continue
            if "extracted_gold" not in record or not record["extracted_gold"]:
                continue
            
            # Parse LLM answer
            llm_text = record["extracted_llm"]
            if is_nested_json_structure(llm_text):
                llm_compounds = parse_nested_json_structure(llm_text)
                llm_items = [comp.get("compound_name", "") for comp in llm_compounds if comp.get("compound_name")]
            else:
                llm_items = parse_list_string(llm_text)
            
            # Parse gold answer
            gold_text = record["extracted_gold"]
            if is_nested_json_structure(gold_text):
                gold_compounds = parse_nested_json_structure(gold_text)
                gold_items = [comp.get("compound_name", "") for comp in gold_compounds if comp.get("compound_name")]
            else:
                gold_items = parse_list_string(gold_text)
            
            # Compare LLM with gold
            llm_score, llm_comparison = compare_llm_with_gold(llm_items, gold_items)
            
            # Add LLM comparison results to record
            record["llm_score"] = round(llm_score, 3)
            record["llm_comparison"] = llm_comparison
            
            # Track LLM statistics
            if isinstance(llm_score, (int, float)):
                llm_scored_records += 1
                llm_total_score += llm_score
                
                if llm_comparison == "match":
                    llm_matches += 1
                elif llm_comparison == "llm_less":
                    llm_less += 1
                elif llm_comparison == "llm_more":
                    llm_more += 1
                elif llm_comparison == "conflict":
                    llm_conflicts += 1
            
            updated_count += 1
            continue
        
        # Reviewer comparison mode (original functionality)
        # Check if extracted_rev1 exists
        if "extracted_rev1" not in record:
            continue
        
        # Check if there's actually a second reviewer (rev2_reviewer not empty)
        if "rev2_reviewer" not in record or not record["rev2_reviewer"] or record["rev2_reviewer"].strip() == "":
            # If no second reviewer, copy rev1 to gold but clean it first
            rev1_text = record["extracted_rev1"]
            if is_nested_json_structure(rev1_text):
                # Handle nested JSON structure
                rev1_compounds = parse_nested_json_structure(rev1_text)
                # Format gold answer to match the input format
                gold_answer = str(rev1_compounds).replace('"', "'")
            else:
                # Handle simple string structure
                rev1_items = parse_list_string(rev1_text)
                # Format gold answer as comma-separated string
                gold_answer = ", ".join(sorted(rev1_items)) if rev1_items else ""
            
            record["extracted_gold"] = gold_answer
            record["rev_score"] = "na"
            updated_count += 1
            continue
        
        # Check data type and parse accordingly
        rev1_text = record["extracted_rev1"]
        rev2_text = record["extracted_rev2"]
        
        # Debug output for first few records
        if processed_count <= 3:
            print(f"DEBUG: Record {processed_count} - rev1 type: {type(rev1_text)}, rev2 type: {type(rev2_text)}")
            print(f"DEBUG: rev1 preview: {str(rev1_text)[:100]}...")
            print(f"DEBUG: rev2 preview: {str(rev2_text)[:100]}...")
            print(f"DEBUG: is_nested_json_structure(rev1): {is_nested_json_structure(rev1_text)}")
            print(f"DEBUG: is_nested_json_structure(rev2): {is_nested_json_structure(rev2_text)}")
        
        if is_nested_json_structure(rev1_text) or is_nested_json_structure(rev2_text):
            print(f"DEBUG: Using NESTED JSON processing for record {processed_count}")
            # Handle nested JSON structure (pesticides, etc.)
            rev1_compounds = parse_nested_json_structure(rev1_text)
            rev2_compounds = parse_nested_json_structure(rev2_text)
            
            print(f"DEBUG: Parsed rev1_compounds: {len(rev1_compounds)} compounds")
            print(f"DEBUG: Parsed rev2_compounds: {len(rev2_compounds)} compounds")
            
            # Compare nested structures
            score, gold_compounds = compare_nested_json_answers(rev1_compounds, rev2_compounds)
            
            print(f"DEBUG: Gold compounds: {len(gold_compounds)} compounds")
            
            # Format gold answer to match the input format (Python dict string)
            # Convert to Python dict string format to maintain consistency
            gold_answer = str(gold_compounds).replace('"', "'")
            
        else:
            print(f"DEBUG: Using SIMPLE processing for record {processed_count}")
            # Handle simple string structure (bee species, etc.)
            rev1_items = parse_list_string(rev1_text)
            rev2_items = parse_list_string(rev2_text)
            
            # Compare simple answers
            score, gold_items = compare_answers(rev1_items, rev2_items)
            
            # Format gold answer as comma-separated string
            gold_answer = ", ".join(sorted(gold_items)) if gold_items else ""
        
        # Update record with reviewer comparison results
        record["extracted_gold"] = gold_answer
        record["rev_score"] = round(score, 3)
        updated_count += 1
        
        # Track statistics (only for records with numerical scores)
        if isinstance(score, (int, float)):
            scored_records += 1
            total_score += score
            
            if score == 1.0:
                perfect_scores += 1
            elif score == 0.0:
                zero_scores += 1
    
    # Calculate and display statistics
    print(f"Updated {updated_count}/{processed_count} records in {file_path}")
    
    # Prepare summary results for saving
    summary_results = {
        "file_path": file_path,
        "method": method,
        "timestamp": datetime.datetime.now().isoformat(),
        "total_records": processed_count,
        "updated_records": updated_count
    }
    
    if method == "llm":
        # Display LLM comparison statistics
        if llm_scored_records > 0:
            avg_score = llm_total_score / llm_scored_records
            print(f"\n--- LLM vs GOLD COMPARISON STATISTICS ---")
            print(f"Total LLM scored records: {llm_scored_records}")
            print(f"LLM average score: {avg_score:.3f}")
            print(f"Exact matches: {llm_matches} ({llm_matches/llm_scored_records*100:.1f}%)")
            print(f"LLM missing information: {llm_less} ({llm_less/llm_scored_records*100:.1f}%)")
            print(f"LLM has extra information: {llm_more} ({llm_more/llm_scored_records*100:.1f}%)")
            print(f"Conflicts: {llm_conflicts} ({llm_conflicts/llm_scored_records*100:.1f}%)")
            
            # Add LLM statistics to summary
            summary_results.update({
                "llm_statistics": {
                    "total_scored_records": llm_scored_records,
                    "average_score": round(avg_score, 3),
                    "exact_matches": llm_matches,
                    "exact_matches_percentage": round(llm_matches/llm_scored_records*100, 1),
                    "llm_missing_info": llm_less,
                    "llm_missing_info_percentage": round(llm_less/llm_scored_records*100, 1),
                    "llm_extra_info": llm_more,
                    "llm_extra_info_percentage": round(llm_more/llm_scored_records*100, 1),
                    "conflicts": llm_conflicts,
                    "conflicts_percentage": round(llm_conflicts/llm_scored_records*100, 1)
                }
            })
        else:
            print(f"No LLM comparison data found.")
            summary_results["llm_statistics"] = None
    else:
        # Display reviewer agreement statistics only
        if scored_records > 0:
            avg_score = total_score / scored_records
            print(f"\n--- REVIEWER AGREEMENT STATISTICS ---")
            print(f"Total scored records: {scored_records}")
            print(f"Perfect agreement (score = 1.0): {perfect_scores} ({perfect_scores/scored_records*100:.1f}%)")
            print(f"No agreement (score = 0.0): {zero_scores} ({zero_scores/scored_records*100:.1f}%)")
            print(f"Average score: {avg_score:.3f}")
            print(f"Records with no second reviewer: {updated_count - scored_records}")
            
            # Add reviewer statistics to summary
            summary_results.update({
                "reviewer_statistics": {
                    "total_scored_records": scored_records,
                    "average_score": round(avg_score, 3),
                    "perfect_agreement": perfect_scores,
                    "perfect_agreement_percentage": round(perfect_scores/scored_records*100, 1),
                    "no_agreement": zero_scores,
                    "no_agreement_percentage": round(zero_scores/scored_records*100, 1),
                    "records_no_second_reviewer": updated_count - scored_records
                }
            })
        else:
            print(f"No records with numerical scores found.")
            summary_results["reviewer_statistics"] = None
    
    # Save summary results to JSON file
    summary_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}-{method}-summary.json"
    
    # Create benchmark_data directory if it doesn't exist
    benchmark_data_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), "llm_benchmarking", "benchmark_data")
    os.makedirs(benchmark_data_dir, exist_ok=True)
    
    summary_filepath = os.path.join(benchmark_data_dir, summary_filename)
    
    try:
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        print(f"\nSummary results saved to: {summary_filepath}")
    except Exception as e:
        print(f"Error saving summary file: {e}")
    
    # Save updated file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing file {file_path}: {e}")


def main():
    """Main function to handle command line arguments and process files."""
    parser = argparse.ArgumentParser(
        description="Compare reviewer answers and create gold answers with scoring, or compare LLM with gold answers"
    )
    parser.add_argument(
        "file_pattern",
        help="File pattern to process (e.g., 'llm_benchmarking/benchmark_data/*_extracted.json')"
    )
    parser.add_argument(
        "--method",
        choices=["reviewers", "llm"],
        default="reviewers",
        help="Comparison method: 'reviewers' to compare rev1 vs rev2 and create gold, 'llm' to compare LLM vs gold (default: reviewers)"
    )
    
    args = parser.parse_args()
    
    if args.method not in ["reviewers", "llm"]:
        print("Only 'reviewers' and 'llm' methods are currently supported")
        return
    
    # Find files matching the pattern
    files = glob.glob(args.file_pattern)
    
    if not files:
        print(f"No files found matching pattern: {args.file_pattern}")
        return
    
    print(f"Found {len(files)} files to process")
    
    # Process each file
    for file_path in sorted(files):
        process_file(file_path, args.method)
    
    print("Processing complete!")


def strip_non_alphanumeric(text: str) -> str:
    """
    Strip all non-alphanumeric characters from text, keeping only letters and numbers.
    This is used for comparison purposes to ignore punctuation, spaces, and special characters.
    """
    if not text:
        return text
    
    # Remove all non-alphanumeric characters (keep only letters and numbers)
    import re
    return re.sub(r'[^a-zA-Z0-9]', '', text.lower())


if __name__ == "__main__":
    main()
