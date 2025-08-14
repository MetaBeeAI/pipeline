
import os
import json
import yaml
import pandas as pd
import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import argparse
import sys
import unidecode  # Add this to your imports

# Load environment variables and initialize client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Directory containing paper folders
BASE_DIR = "data/papers"

def load_schema_config(config_file="../schema_config.yaml"):
    """Load the schema configuration from a YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def extract_with_wildcards(data, paths):
    """Extract text from nested paths in data, with support for wildcards."""
    extracted_texts = []
    
    for path in paths:
        # Start with the full data
        results = _extract_path_data(data, path)
        extracted_texts.extend(results)
    
    if extracted_texts:
        return " ".join(extracted_texts)
    return ""

def _extract_path_data(data, path, path_index=0):
    """Recursively extract data from a path, handling wildcards."""
    # Base case: we've reached the end of the path
    if path_index >= len(path):
        if isinstance(data, str):
            return [data]
        elif data is not None:
            return [str(data)]
        return []
    
    results = []
    current_key = path[path_index]
    
    # Handle wildcard segment
    if current_key == "*":
        # If we're at a wildcard, iterate through all keys at this level
        if isinstance(data, dict):
            for value in data.values():
                # Continue traversing with each value
                results.extend(_extract_path_data(value, path, path_index + 1))
        elif isinstance(data, list):
            for item in data:
                # Continue traversing with each item
                results.extend(_extract_path_data(item, path, path_index + 1))
    # Handle normal segment
    elif isinstance(data, dict) and current_key in data:
        # Continue traversing with the value for this key
        results.extend(_extract_path_data(data[current_key], path, path_index + 1))
    
    return results

def extract_structured_data(text, field_config, existing_data=None):
    """Extract structured data based on the field configuration."""
    field_name = field_config['field']
    field_type = field_config['type']
    field_description = field_config.get('description', '')
    
    # Handle complex type with subfields (structured relationships)
    if field_type in ['complex', 'complex_array']:
        subfields = field_config.get('subfields', [])
        if not subfields:
            print(f"Warning: Complex field {field_name} has no subfields defined")
            return {} if field_type == 'complex' else []
        
        # For complex fields, we need a different prompt strategy
        if field_type == 'complex_array':
            # This is for a nested array inside a complex type
            prompt = (
                f"Extract information about {field_name} from the following text. {field_description}\n\n"
                f"Text: {text}\n\n"
                "Return a JSON array of objects, where each object represents one instance with the following properties:\n"
            )
        else:
            prompt = (
                f"Extract structured information from the following text about {field_name}. {field_description}\n\n"
                f"Text: {text}\n\n"
                "Return a JSON array of objects, where each object represents a distinct item with the following properties:\n"
            )
        
        # Add each subfield to the prompt
        for subfield in subfields:
            subfield_name = subfield['field']
            subfield_type = subfield['type']
            subfield_desc = subfield.get('description', '')
            
            if subfield_type == 'complex_array':
                # For nested complex arrays, show a more detailed structure
                nested_subfields = subfield.get('subfields', [])
                prompt += f"- {subfield_name}: Array of objects, each with: {subfield_desc}\n"
                
                # Show the structure of each sub-object
                for nested_field in nested_subfields:
                    nested_name = nested_field['field']
                    nested_type = nested_field['type']
                    nested_desc = nested_field.get('description', '')
                    prompt += f"  * {nested_name}: {nested_desc} ({nested_type})\n"
            else:
                prompt += f"- {subfield_name}: {subfield_desc} ({subfield_type})\n"
        
        # Build a thorough example
        example = {}
        for subfield in subfields:
            if subfield['type'] == 'complex_array':
                # Create an example of a nested array
                nested_example = {}
                for nested_field in subfield.get('subfields', []):
                    if nested_field['type'] == 'list':
                        if nested_field.get('number', False):
                            nested_example[nested_field['field']] = [1.0, 2.5, 5.0]
                        else:
                            nested_example[nested_field['field']] = ["example1", "example2"]
                    elif nested_field['type'] == 'number':
                        nested_example[nested_field['field']] = 5.2
                    else:
                        nested_example[nested_field['field']] = "example value"
                
                example[subfield['field']] = [nested_example]
            elif subfield['type'] == 'list':
                if subfield.get('number', False):
                    example[subfield['field']] = [1.0, 2.5, 5.0]
                else:
                    example[subfield['field']] = ["example1", "example2"]
            elif subfield['type'] == 'number':
                example[subfield['field']] = 5.2
            else:
                example[subfield['field']] = "example value"
        
        # Provide a more specific example for different field types
        if field_type == 'complex_array':
            prompt += f"\nExample output format: {json.dumps([example])}"
        else:
            # For pesticides_data, provide a more specific example
            if field_name == 'pesticides_data':
                detailed_example = [
                    {
                        "name": "imidacloprid",
                        "exposure_methods": [
                            {
                                "method": "oral",
                                "doses": [0.1, 1.0, 10.0],
                                "dose_units": "ng/bee",
                                "exposure_duration": [48, 72],
                                "exposure_units": "hours"
                            },
                            {
                                "method": "contact",
                                "doses": [0.5, 5.0],
                                "dose_units": "μg/bee",
                                "exposure_duration": [24],
                                "exposure_units": "hours"
                            }
                        ]
                    },
                    {
                        "name": "clothianidin",
                        "exposure_methods": [
                            {
                                "method": "oral",
                                "doses": [0.2, 2.0],
                                "dose_units": "ng/bee",
                                "exposure_duration": [48],
                                "exposure_units": "hours"
                            }
                        ]
                    }
                ]
                prompt += f"\nExample output format for pesticides with multiple exposure methods: {json.dumps(detailed_example)}"
            elif field_name == 'endpoints':
                # Add a detailed, realistic example for endpoints
                endpoints_example = [
                    {
                        "endpoint": "mortality",
                        "effect_direction": {
                            "significance": "significant",
                            "direction": "negative"
                        },
                        "sample_size": {
                            "num": 30,
                            "units": "bees"
                        },
                        "measurement_type": {
                            "central_tendency_type": "mean",
                            "variability_type": "standard error"
                        },
                        "measurements": [
                            {
                                "treatment": "control",
                                "central_tendency": 5.2,
                                "variability": 1.1,
                                "units": "percent"
                            },
                            {
                                "treatment": "imidacloprid 5ng/bee",
                                "central_tendency": 38.7,
                                "variability": 3.2,
                                "units": "percent"
                            }
                        ],
                        "pesticide": "imidacloprid",
                        "exposure_method": "oral"
                    },
                    {
                        "endpoint": "learning performance",
                        "effect_direction": {
                            "significance": "significant",
                            "direction": "negative"
                        },
                        "sample_size": {
                            "num": 25,
                            "units": "bees per treatment"
                        },
                        "measurement_type": {
                            "central_tendency_type": "mean",
                            "variability_type": "standard deviation"
                        },
                        "measurements": [
                            {
                                "treatment": "control",
                                "central_tendency": 87.3,
                                "variability": 5.6,
                                "units": "percent correct responses"
                            },
                            {
                                "treatment": "clothianidin 0.2ng/bee",
                                "central_tendency": 62.1,
                                "variability": 7.9,
                                "units": "percent correct responses"
                            }
                        ],
                        "pesticide": "clothianidin",
                        "exposure_method": "oral"
                    }
                ]
                prompt += f"\nExample output format for detailed endpoint measurements: {json.dumps(endpoints_example, indent=2)}"
            else:
                prompt += f"\nExample output format: {json.dumps([example])}"
                
                # Build exclusion list if needed
                exclude_fields = field_config.get('exclude_fields', [])
                if exclude_fields and existing_data:
                    exclusions = []
                    for exclude_field in exclude_fields:
                        # Handle dot notation for nested fields
                        if '.' in exclude_field and existing_data:
                            parts = exclude_field.split('.')
                            if parts[0] in existing_data:
                                nested_data = existing_data[parts[0]]
                                if isinstance(nested_data, list):
                                    for item in nested_data:
                                        if parts[1] in item:
                                            exclusions.append(item[parts[1]])
                        # Handle direct fields
                        elif exclude_field in existing_data and existing_data[exclude_field]:
                            if isinstance(existing_data[exclude_field], list):
                                exclusions.extend(existing_data[exclude_field])
                            else:
                                exclusions.append(str(existing_data[exclude_field]))
                    
                    if exclusions:
                        exclusion_list = '", "'.join(str(item) for item in exclusions)
                        prompt += f'\n\nExclude the following items that have already been identified: "{exclusion_list}".'
            
    else:
        # Original logic for simple types
        is_number = field_config.get('number', False)
        exclude_fields = field_config.get('exclude_fields', [])
        
        # Build the exclusion information
        exclusion_text = ""
        if exclude_fields and existing_data:
            exclusions = []
            for exclude_field in exclude_fields:
                if '.' in exclude_field and existing_data:
                    parts = exclude_field.split('.')
                    if parts[0] in existing_data:
                        nested_data = existing_data[parts[0]]
                        if isinstance(nested_data, list):
                            for item in nested_data:
                                if parts[1] in item:
                                    exclusions.append(item[parts[1]])
                elif exclude_field in existing_data and existing_data[exclude_field]:
                    if isinstance(existing_data[exclude_field], list):
                        exclusions.extend(existing_data[exclude_field])
                    else:
                        exclusions.append(str(existing_data[exclude_field]))
            
            if exclusions:
                exclusion_list = '", "'.join(str(item) for item in exclusions)
                exclusion_text = f' Exclude the following items that have already been identified: "{exclusion_list}".'
        
        prompt = (
            f"Extract the {field_name} from the following text. {field_description}{exclusion_text}\n\n"
            f"Text: {text}\n\n"
            f"The {field_name} should be formatted as a {field_type}."
        )
        
        if field_type == 'list':
            prompt += " Return a JSON array like [\"item1\", \"item2\", ...]"
            if is_number:
                prompt += " with numeric values instead of strings."
        elif field_type == 'number':
            prompt += " Return only the number as a JSON value (e.g., 5.2)."
    
    # Make the API call with our prompt
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a data extraction assistant. Return only valid JSON with no additional text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    # Get the raw response content
    raw_content = response.choices[0].message.content.strip()
    print(f"Extracting {field_name}: {raw_content}")
    
    # Try to find JSON in the response text
    try:
        # First try direct parsing
        result = json.loads(raw_content)
        return result
    except json.JSONDecodeError:
        # Try to extract JSON by looking for patterns based on the expected type
        try:
            if field_type == 'complex':
                # Look for array of objects
                start_idx = raw_content.find('[')
                end_idx = raw_content.rfind(']') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = raw_content[start_idx:end_idx]
                    return json.loads(json_str)
                return []
            elif field_type == 'list':
                # Look for square brackets
                start_idx = raw_content.find('[')
                end_idx = raw_content.rfind(']') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = raw_content[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    # If no brackets, try splitting by commas and cleaning up
                    items = [item.strip().strip('"\'') for item in raw_content.split(',')]
                    return items
            elif field_type == 'number':
                # Extract just the number
                import re
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", raw_content)
                if numbers:
                    return float(numbers[0])
                return None
            else:
                # For other types, just return as string
                return raw_content
        except Exception as e:
            print(f"Extraction failed for {field_name}: {e}")
            # Return appropriate default value based on type
            if field_type == 'complex':
                return []
            elif field_type == 'list':
                return []
            elif field_type == 'number':
                return None
            else:
                return ""

def process_complex_field(data, field_config, parent_text=None, existing_data=None):
    """Process a complex field with nested data."""
    field_name = field_config['field']
    subfields = field_config.get('subfields', [])
    
    # Special handling for endpoints
    if field_name == 'endpoints':
        # Use the specialized endpoint processor
        return process_endpoints(data, field_config, parent_text)
    
    # Extract data using paths defined in this field
    field_paths = field_config.get('paths', [])
    extracted_text = ""
    
    if field_paths:
        extracted_text = extract_with_wildcards(data, field_paths)
    
    # If we have no text from paths but have parent text, use that
    if not extracted_text and parent_text:
        extracted_text = parent_text
    
    # If we have text to process, extract the data
    if extracted_text:
        return extract_structured_data(extracted_text, field_config, existing_data)
    
    # If no text was found but we have subfields, create an empty structure
    return {} if field_config['type'] == 'complex' else []

def process_endpoints_independently(data, field_config, parent_data=None):
    """Process endpoints as a separate top-level entity."""
    endpoints = []
    
    print(f"Starting endpoint processing for paper {parent_data.get('PaperID', 'Unknown')}")
    
    # Navigate to the correct endpoint list structure
    # It should be under QUESTIONS -> endpoint -> list
    if "QUESTIONS" in data and "endpoint" in data["QUESTIONS"] and "list" in data["QUESTIONS"]["endpoint"]:
        endpoint_list = data["QUESTIONS"]["endpoint"]["list"]
        print(f"Found {len(endpoint_list)} endpoints to process")
        
        # Get paper ID for reference
        paper_id = parent_data.get("PaperID") if parent_data else "Unknown"
        
        # Record pesticides for potential cross-referencing
        pesticides = []
        if "pesticides_data" in parent_data and parent_data["pesticides_data"]:
            for pesticide in parent_data["pesticides_data"]:
                if "name" in pesticide:
                    pesticides.append(pesticide["name"])
        
        # Iterate through each endpoint
        for endpoint_name, endpoint_data in endpoint_list.items():
            print(f"Processing endpoint: {endpoint_name}")
            # Check if there's a results section with answer/reason
            if "results" in endpoint_data:
                # Gather text from this specific endpoint
                result_data = endpoint_data["results"]
                texts = []
                
                if "answer" in result_data:
                    texts.append(result_data["answer"])
                if "reason" in result_data:
                    texts.append(result_data["reason"])
                
                # If we found text for this endpoint
                if texts:
                    endpoint_text = " ".join(texts)
                    
                    # Create base endpoint object with paper ID and name
                    base_endpoint = {
                        "paper_id": paper_id,
                        "endpoint": endpoint_name
                    }
                    
                    # Extract endpoint details using our schema
                    subfields = field_config.get('subfields', [])
                    prompt = (
                        f"Extract details about the endpoint '{endpoint_name}' from this text. Include sample size, effect direction, "
                        f"central tendency (mean), variability, and effect units if available.\n\n"
                        f"Text: {endpoint_text}\n\n"
                        "Return a JSON object with these properties:\n"
                    )
                    
                    for subfield in subfields:
                        sf_name = subfield['field']
                        sf_desc = subfield.get('description', '')
                        prompt += f"- {sf_name}: {sf_desc}\n"
                    
                    # Call the OpenAI API
                    try:
                        print(f"  Calling OpenAI API for endpoint '{endpoint_name}'")
                        response = client.chat.completions.create(
                            model="gpt-4-turbo-preview",
                            temperature=0.0,
                            messages=[
                                {"role": "system", "content": "You are a helpful research assistant that extracts structured information from text."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        
                        # Parse the response
                        result_text = response.choices[0].message.content.strip()
                        
                        # Extract the JSON part from the response
                        import re
                        json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
                        if json_match:
                            result_text = json_match.group(1)
                        else:
                            # Try to find JSON without code block
                            json_match = re.search(r'(\{.*\})', result_text, re.DOTALL)
                            if json_match:
                                result_text = json_match.group(1)
                        
                        try:
                            # Parse the JSON
                            endpoint_info = json.loads(result_text)
                            print(f"  Successfully extracted data for endpoint '{endpoint_name}'")
                            
                            # Combine with base endpoint data
                            complete_endpoint = base_endpoint.copy()
                            
                            # Add extracted fields to the endpoint
                            for key, value in endpoint_info.items():
                                complete_endpoint[key] = value
                            
                            # Try to infer pesticide or stressor if it's in the endpoint name but not extracted
                            if ('pesticide' not in complete_endpoint or not complete_endpoint['pesticide']) and pesticides:
                                for pesticide in pesticides:
                                    if pesticide.lower() in endpoint_name.lower():
                                        complete_endpoint['pesticide'] = pesticide
                                        break
                            
                            endpoints.append(complete_endpoint)
                        except json.JSONDecodeError as e:
                            # Fallback to just using the name and paper ID
                            print(f"  Failed to parse endpoint JSON for {endpoint_name} in {paper_id}: {e}")
                            print(f"  Response text: {result_text}")
                            endpoints.append(base_endpoint)
                    
                    except Exception as e:
                        print(f"  Error processing endpoint {endpoint_name}: {str(e)}")
                        endpoints.append(base_endpoint)
            else:
                print(f"  No results section found for endpoint: {endpoint_name}")
    else:
        print(f"No endpoint list structure found in data for paper {parent_data.get('PaperID', 'Unknown')}")
    
    print(f"Completed endpoint processing, found {len(endpoints)} endpoints")
    return endpoints

def process_papers(start_folder, end_folder, config_file="../schema_config.yaml", 
                   process_pesticides=True, process_endpoints=True):
    """
    Process papers using the schema configuration.
    
    Args:
        start_folder: First folder number to process
        end_folder: Last folder number to process
        config_file: Path to schema configuration file
        process_pesticides: Whether to process pesticide data
        process_endpoints: Whether to process endpoint data
    
    Returns:
        Tuple of (pesticide_results, endpoint_results)
    """
    # Load the schema configuration
    config = load_schema_config(config_file)
    output_schema = config['output_schema']
    data_mappings = config['data_mappings']
    
    # Initialize results containers
    pesticide_results = []
    endpoint_results = []
    
    for i in range(start_folder, end_folder + 1):
        folder_name = f"{i:03d}"  # Convert to three-digit format
        folder = Path(BASE_DIR) / folder_name
        
        if folder.is_dir():
            paper_id = folder.name  # Extract folder name as PaperID
            json_path = folder / "answers.json"
            
            if json_path.exists():
                print(f"\nProcessing paper {paper_id}...")
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Only create and process the structured row if we're processing pesticides
                if process_pesticides:
                    structured_row = {"PaperID": paper_id}
                    
                    # Process each field defined in the output schema except endpoints
                    for field_config in output_schema:
                        field_name = field_config['field']
                        
                        # Skip endpoints - they're handled separately
                        if field_name == 'endpoints':
                            continue
                        
                        # Get the mapping for this field
                        mapping = data_mappings.get(field_name, {})
                        
                        # Try to extract using paths first
                        field_paths = field_config.get('paths', mapping.get('paths', []))
                        if field_paths:
                            field_text = extract_with_wildcards(data, field_paths)
                            if field_text:
                                field_value = extract_structured_data(field_text, field_config, structured_row)
                                structured_row[field_name] = field_value
                        # Fall back to category/question approach
                        elif 'category' in mapping and 'question' in mapping:
                            category = mapping['category']
                            question = mapping['question']
                            
                            # Find the answer in the data
                            if "QUESTIONS" in data and category in data["QUESTIONS"]:
                                for q_key, q_value in data["QUESTIONS"][category].items():
                                    if isinstance(q_value, dict) and "answer" in q_value:
                                        # Extract the structured data for this field
                                        field_value = extract_structured_data(q_value["answer"], field_config, structured_row)
                                        structured_row[field_name] = field_value
                                        break
                    
                    # Add to pesticide results
                    pesticide_results.append(structured_row)
                
                # Only process endpoints if requested
                if process_endpoints:
                    # Need paper_id for endpoint processing
                    base_data = {"PaperID": paper_id}
                    
                    # If we already processed pesticides, use that data for context
                    if process_pesticides:
                        base_data = structured_row
                    
                    # Now process endpoints separately
                    for field_config in output_schema:
                        if field_config['field'] == 'endpoints':
                            paper_endpoints = process_endpoints_independently(data, field_config, base_data)
                            
                            # Add paper_id to each endpoint if not already present
                            for endpoint in paper_endpoints:
                                if 'paper_id' not in endpoint:
                                    endpoint['paper_id'] = paper_id
                            
                            endpoint_results.extend(paper_endpoints)
    
    return pesticide_results, endpoint_results

def clean_text_for_csv(text):
    """Clean text for CSV output by removing brackets, quotes, and transliterating Unicode characters."""
    if not isinstance(text, str):
        return text
    
    # Replace square brackets
    text = text.replace('[', '').replace(']', '')
    
    # Replace single quotes (but be careful with apostrophes in words)
    text = text.replace("'", "")
    
    # Convert all Unicode to ASCII equivalents (handles any symbol regardless of representation)
    text = unidecode.unidecode(text)
    
    # Handle specific scientific notations that need manual correction
    # These are cases where the transliteration might not be intuitive
    corrections = {
        "ug/": "µg/",     # Make sure micrograms are clear
        "a.i./uL": "a.i./µL",  # Active ingredient per microliter
        "+/-": "±",       # Common in scientific notation
        "deg C": "°C",    # Temperature
        "x10": "×10",     # Scientific notation 
    }
    
    # Replace the corrections with the specialized scientific notation for readability
    # This is for CSV column headers which need to be descriptive
    for plain, scientific in corrections.items():
        if plain in text:
            text = text.replace(plain, f"{plain} ({scientific})")
    
    return text

def clean_data_for_csv(data):
    """Recursively clean all string values in a nested data structure."""
    if isinstance(data, str):
        return clean_text_for_csv(data)
    elif isinstance(data, list):
        return [clean_data_for_csv(item) for item in data]
    elif isinstance(data, dict):
        return {k: clean_data_for_csv(v) for k, v in data.items()}
    else:
        return data

def save_data(results, filename_prefix, output_dir="output", flatten_func=None):
    """Save data to both JSON and CSV formats with a consistent approach."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON (keep original format with symbols)
    json_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"{filename_prefix.capitalize()} data saved to {json_path}")
    
    # Save CSV with cleaned data
    csv_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.csv")
    
    # Apply flattening if provided, otherwise use results directly
    if flatten_func:
        csv_data = flatten_func(results)
    else:
        csv_data = results.copy()
    
    # Clean the data for CSV
    cleaned_csv_data = clean_data_for_csv(csv_data)
    
    df = pd.DataFrame(cleaned_csv_data)
    df.to_csv(csv_path, index=False)
    print(f"{filename_prefix.capitalize()} data saved to {csv_path}")

def flatten_pesticide_data(pesticide_results):
    """Flatten pesticide data for CSV output."""
    flat_results = []
    for paper in pesticide_results:
        paper_id = paper['PaperID']
        
        # Create a base row with non-complex fields
        base_row = {'PaperID': paper_id}
        for key, value in paper.items():
            if key != 'PaperID' and key != 'pesticides_data':
                if isinstance(value, dict) and key in value:
                    # Handle nested values like {'bee_species': ['species1', 'species2']}
                    if isinstance(value[key], list):
                        # Clean each item before joining
                        cleaned_items = [clean_text_for_csv(str(item)) for item in value[key]]
                        base_row[key] = ', '.join(cleaned_items)
                    else:
                        base_row[key] = clean_text_for_csv(str(value[key]))
                else:
                    base_row[key] = clean_text_for_csv(str(value)) if value is not None else None
        
        # Handle complex pesticides_data
        if 'pesticides_data' in paper and paper['pesticides_data']:
            for pesticide in paper['pesticides_data']:
                pesticide_name = pesticide.get('name', 'Unknown')
                pesticide_name = clean_text_for_csv(pesticide_name)
                
                # If pesticide has exposure methods
                if 'exposure_methods' in pesticide and pesticide['exposure_methods']:
                    for method in pesticide['exposure_methods']:
                        # Create one row per exposure method
                        row = base_row.copy()
                        row['pesticide_name'] = pesticide_name
                        
                        # Add exposure method details
                        method_name = method.get('method', 'Unknown')
                        row['exposure_method'] = clean_text_for_csv(method_name)
                        
                        # Add other method details
                        for key, value in method.items():
                            if key != 'method':
                                if isinstance(value, list):
                                    # Clean each item before joining
                                    cleaned_items = [clean_text_for_csv(str(item)) for item in value]
                                    row[f'method_{key}'] = ', '.join(cleaned_items)
                                else:
                                    row[f'method_{key}'] = clean_text_for_csv(str(value)) if value is not None else None
                        
                        flat_results.append(row)
                else:
                    # No exposure methods, just create one row for this pesticide
                    row = base_row.copy()
                    row['pesticide_name'] = pesticide_name
                    flat_results.append(row)
        else:
            # No pesticides, just use the base row
            flat_results.append(base_row)
    
    return flat_results

if __name__ == "__main__":
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        # Use argparse as defined above
        parser = argparse.ArgumentParser(description="Process paper data to extract pesticides and/or endpoints.")
        parser.add_argument("--start", type=int, required=True, help="Start folder number (e.g., 1 for 001)")
        parser.add_argument("--end", type=int, required=True, help="End folder number (e.g., 4 for 004)")
        parser.add_argument("--config", type=str, default="../schema_config.yaml", help="Path to config file")
        parser.add_argument("--pesticides", action="store_true", help="Process pesticide data")
        parser.add_argument("--endpoints", action="store_true", help="Process endpoint data")
        parser.add_argument("--all", action="store_true", help="Process both pesticide and endpoint data")
        
        args = parser.parse_args()
        
        # Determine what to process
        process_pesticides = args.pesticides or args.all
        process_endpoints = args.endpoints or args.all
        
        # If neither is specified, process both
        if not process_pesticides and not process_endpoints:
            process_pesticides = True
            process_endpoints = True
            
        # Process the papers selectively
        pesticide_results, endpoint_results = process_papers(
            args.start, 
            args.end, 
            args.config,
            process_pesticides=process_pesticides,
            process_endpoints=process_endpoints
        )
        
        # Save the results as appropriate
        if process_pesticides and pesticide_results:
            save_data(pesticide_results, "pesticides", flatten_func=flatten_pesticide_data)
        
        if process_endpoints and endpoint_results:
            save_data(endpoint_results, "endpoints")
    else:
        # Use interactive mode
        start_folder = int(input("Enter start folder number (e.g., 1 for 001): "))
        end_folder = int(input("Enter end folder number (e.g., 4 for 004): "))
        config_file = input("Enter path to config file (default: ../schema_config.yaml): ") or "schema_config.yaml"
        
        # Ask which types to process
        print("What would you like to process?")
        print("1. Pesticides only")
        print("2. Endpoints only")
        print("3. Both pesticides and endpoints")
        choice = input("Enter your choice (1-3): ")
        
        process_pesticides = choice in ['1', '3']
        process_endpoints = choice in ['2', '3']
        
        # Process the papers selectively
        pesticide_results, endpoint_results = process_papers(
            start_folder, 
            end_folder, 
            config_file,
            process_pesticides=process_pesticides,
            process_endpoints=process_endpoints
        )
        
        # Save the results as appropriate
        if process_pesticides and pesticide_results:
            save_data(pesticide_results, "pesticides", flatten_func=flatten_pesticide_data)
        
        if process_endpoints and endpoint_results:
            save_data(endpoint_results, "endpoints")
