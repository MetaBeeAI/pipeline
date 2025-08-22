import json
import asyncio
import logging
import time
from pprint import pprint
from typing import List, Dict, Any, Callable
from tqdm import tqdm  # progress bar for loops
import litellm
from litellm import acompletion
from pydantic import BaseModel

# Configure logging for debugging and error tracking.
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

RETRY = 100

# Load questions configuration from YAML file
import yaml
import os

def load_questions_config():
    """Load questions configuration from the YAML file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    questions_path = os.path.join(script_dir, 'questions.yml')
    
    try:
        with open(questions_path, 'r') as file:
            config = yaml.safe_load(file)
        return config.get('QUESTIONS', {})
    except Exception as e:
        logger.error(f"Error loading questions config: {e}")
        # Return default configuration if YAML loading fails
        return {}

# Load the configuration
QUESTIONS_CONFIG = load_questions_config()

def get_question_config(question_text: str) -> dict:
    """
    Get configuration for a specific question type based on the question text.
    
    Args:
        question_text (str): The question text to analyze.
        
    Returns:
        dict: Configuration with max_chunks, min_score, description, and no_info_response.
    """
    question_lower = question_text.lower()
    
    # Try to find a matching question in the YAML configuration
    for question_key, question_config in QUESTIONS_CONFIG.items():
        if question_config.get('question', '').lower() in question_lower or question_key.lower() in question_lower:
            # Extract configuration from YAML
            config = {
                'max_chunks': question_config.get('max_chunks', 5),
                'min_score': question_config.get('min_score', 0.4),
                'description': question_config.get('description', 'Default configuration'),
                'no_info_response': question_config.get('no_info_response', 'Information not found in the provided text.')
            }
            logger.info(f"Found question config for '{question_key}': {config}")
            return config
    
    # If no exact match found, try keyword-based matching
    if any(word in question_lower for word in ['species', 'bee', 'apis', 'bombus']):
        return get_default_config('bee_species')
    elif any(word in question_lower for word in ['pesticide', 'chemical', 'dose', 'exposure']):
        return get_default_config('pesticides')
    elif any(word in question_lower for word in ['stressor', 'temperature', 'parasite', 'pathogen']):
        return get_default_config('additional_stressors')
    elif any(word in question_lower for word in ['method', 'experiment', 'trial', 'procedure']):
        return get_default_config('experimental_methodology')
    elif any(word in question_lower for word in ['finding', 'result', 'effect', 'impact']):
        return get_default_config('significance')
    elif any(word in question_lower for word in ['future', 'research', 'next', 'suggest']):
        return get_default_config('future_research')
    else:
        # Default configuration
        return {
            'max_chunks': 5,
            'min_score': 0.4,
            'description': 'Default configuration for general questions',
            'no_info_response': 'Information not found in the provided text.'
        }

def get_default_config(question_type: str) -> dict:
    """
    Get default configuration for a question type if YAML loading fails.
    
    Args:
        question_type (str): The type of question.
        
    Returns:
        dict: Default configuration.
    """
    default_configs = {
        'bee_species': {'max_chunks': 3, 'min_score': 0.6, 'description': 'High threshold - species should be explicitly stated', 'no_info_response': 'Species not specified'},
        'pesticides': {'max_chunks': 5, 'min_score': 0.5, 'description': 'Medium threshold - look for chemical names, doses, methods', 'no_info_response': 'No pesticides were tested in this study'},
        'additional_stressors': {'max_chunks': 4, 'min_score': 0.5, 'description': 'Medium threshold - look for non-pesticide stressors', 'no_info_response': 'No additional stressors were tested'},
        'experimental_methodology': {'max_chunks': 5, 'min_score': 0.4, 'description': 'Lower threshold - methods can be spread across sections', 'no_info_response': 'Experimental methodology not clearly described in this study'},
        'significance': {'max_chunks': 4, 'min_score': 0.4, 'description': 'Lower threshold - findings can be in results or discussion', 'no_info_response': 'No specific findings regarding pesticide effects were reported in this study'},
        'future_research': {'max_chunks': 3, 'min_score': 0.4, 'description': 'Lower threshold - future work often in discussion', 'no_info_response': 'No future research directions were suggested'}
    }
    return default_configs.get(question_type, {'max_chunks': 5, 'min_score': 0.4, 'description': 'Default configuration', 'no_info_response': 'Information not found in the provided text.'})


# --------------------------------------------------------------------------
# Data Models using Pydantic for response validation
# --------------------------------------------------------------------------

class Relevance(BaseModel):
    """
    Model for representing the relevance response.

    Attributes:
        reason (str): Explanation for the relevance decision.
        relevance (bool): Flag indicating whether the text is relevant to the question.
        is_bib_list (bool): Flag indicating whether the text is a bibliography list.
        relevance_score (float): Numeric relevance score from 0.0 to 1.0.
    """
    reason: str
    relevance: bool
    is_bib_list: bool
    relevance_score: float


class Answer(BaseModel):
    """
    Model for representing the answer response.

    Attributes:
        reason (str): Explanation or reasoning behind the answer.
        answer (str): The answer to the provided question.
    """
    reason: str
    answer: str

class AnswerWithChunkId(BaseModel):
    """
    Model for representing the answer response.

    Attributes:
        reason (str): Explanation or reasoning behind the answer.
        answer (str): The answer to the provided question.
        chunk_ids (List[str]): The chunk id of the text that was used to generate the answer.
    """
    reason: str
    answer: str
    chunk_ids: List[str]

class AnswerList(BaseModel):
    """
    Model for representing the answer response.

    Attributes:
        reason (str): Explanation or reasoning behind the answer.
        answer (List[str]): The answer to the provided question.
    """
    answer: List[str]

# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------

def load_json_file(path: str) -> Dict[str, Any]:
    """
    Load a JSON file from the given file path.

    Args:
        path (str): The file path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON object.
    """
    with open(path, 'r') as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Asynchronous Processing Functions
# --------------------------------------------------------------------------


async def format_to_list(question,text, model: str = 'openai/gpt-4o-mini') -> Dict[str, Any]:
    """
    Retrieve an answer for the given question using the provided text chunk.

    This function constructs a prompt by embedding the question and text chunk,
    calls the asynchronous API to obtain an answer, and adds the parsed answer
    to the chunk dictionary.

    Args:
        question (str): The question for which the answer is sought.
        chunk (Dict[str, Any]): Dictionary containing the text and related data.
        model (str, optional): Model identifier for the API call. Defaults to 'openai/gpt-4o'.

    Returns:
        Dict[str, Any]: Updated chunk with an added 'answer' field containing the response.
    """
    prompt: str = f"""
Please evaluate the following question and the answer text to the question. The answer should contain one or more entities.
Return the entities in a list format.

<Question>
{question}
</Question>

<Text>
{text}
</Text>
    """.strip()

    messages = [{"content": prompt, "role": "user"}]

    result = None
    for i in range(RETRY):
        try:
            # Call the API asynchronously expecting a response conforming to the Answer model.
            response = await acompletion(model=model, messages=messages, response_format=AnswerList,temperature=0)
            # Parse the JSON string from the API response.
            result = json.loads(response.choices[0].message.content)
            logger.info("Answer restructured", result)
            break
        except Exception as e:
            logger.error("Error obtaining answer restructuring",  e)
            time.sleep(1)
            continue
    return result


async def get_answer(question: str, chunk: Dict[str, Any], model: str = 'openai/gpt-4o-mini') -> Dict[str, Any]:
    """
    Retrieve an answer for the given question using the provided text chunk.

    This function constructs a prompt by embedding the question and text chunk,
    calls the asynchronous API to obtain an answer, and adds the parsed answer
    to the chunk dictionary.

    Args:
        question (str): The question for which the answer is sought.
        chunk (Dict[str, Any]): Dictionary containing the text and related data.
        model (str, optional): Model identifier for the API call. Defaults to 'openai/gpt-4o'.

    Returns:
        Dict[str, Any]: Updated chunk with an added 'answer' field containing the response.
    """
    text: str = chunk.get('text', '')
    prompt: str = f"""
Please evaluate the following text and return the answer to the question.

<Question>
{question}
</Question>

<Text>
{text}
</Text>
    """.strip()

    messages = [{"content": prompt, "role": "user"}]

    for i in range(RETRY):
        try:
            # Call the API asynchronously expecting a response conforming to the Answer model.
            response = await acompletion(model=model, messages=messages, response_format=Answer,temperature=0)
            # Parse the JSON string from the API response.
            chunk['answer'] = json.loads(response.choices[0].message.content)
            logger.info("Answer obtained for chunk %s: %s", chunk.get('chunk_id'), chunk['answer'])
            break
        except Exception as e:
            logger.error("Error obtaining answer for chunk %s: %s", chunk.get('chunk_id'), e)
            chunk['answer'] = None  # In case of error, mark answer as None.
            time.sleep(1)
            continue
    return chunk


async def check_relevance(question: str, chunk: Dict[str, Any], model: str = 'openai/gpt-4o-mini') -> Dict[str, Any]:
    """
    Check if the provided text chunk is relevant to the given question.

    Constructs a prompt combining the question with the text chunk, calls the API,
    and attaches the relevance result to the chunk dictionary.

    Args:
        question (str): The question to test relevance against.
        chunk (Dict[str, Any]): Dictionary containing the text to be evaluated.
        model (str, optional): Model identifier for the API call. Defaults to 'openai/gpt-4o'.

    Returns:
        Dict[str, Any]: Updated chunk with an added 'relevance' field containing the API response.
    """
    text: str = chunk.get('text', '')
    prompt: str = f"""
Please evaluate the following text and indicate whether it is relevant to the question. 
Please evaluate if the text is a reference / citation / bibliography list.

<Question>
{question}
</Question>

<Text>
{text}
</Text>

Please provide:
1. A relevance score from 0.0 to 1.0 where:
   - 0.0 = Completely irrelevant or no useful information
   - 0.3 = Slightly relevant, minimal useful information
   - 0.5 = Moderately relevant, some useful information
   - 0.7 = Highly relevant, substantial useful information
   - 1.0 = Extremely relevant, directly answers the question
2. A brief explanation for your relevance decision
3. Whether this is a bibliography/reference list
    """.strip()

    messages = [{"content": prompt, "role": "user"}]

    for i in range(RETRY):
        try:
            response = await acompletion(model=model, messages=messages, response_format=Relevance,temperature=0)
            outcome = json.loads(response.choices[0].message.content)

            chunk['relevance'] = outcome
            chunk['relevance_score'] = outcome.get('relevance_score', 0.0) # Extract and assign score

            if outcome.get('is_bib_list'):
                logger.info("Detected bibliography list for chunk %s", chunk.get('chunk_id'))
                # Mark the chunk as irrelevant if it is a bibliography list.
                chunk['relevance'] = {'relevance': False, 'reason': 'Detected as bibliography list.'}
                chunk['relevance_score'] = 0.0 # Set score to 0 for irrelevant chunks

            logger.info("Relevance for chunk %s: %s", chunk.get('chunk_id'), outcome)
            break
        except Exception as e:
            logger.error("Error checking relevance for chunk %s: %s", chunk.get('chunk_id'), e)
            # Mark the chunk as irrelevant in case of any error.
            chunk['relevance'] = {'relevance': False, 'reason': str(e)}
            chunk['relevance_score'] = 0.0 # Set score to 0 for irrelevant chunks
            time.sleep(1)
            continue
    return chunk


async def filter_all_chunks(question: str, chunks: List[Dict[str, Any]], max_chunks: int = 5, min_score: float = 0.3) -> List[Dict[str, Any]]:
    """
    Filter a list of text chunks to retain only those relevant to the question.
    Limits results to top N most relevant chunks above a minimum score.

    Args:
        question (str): The question used for relevance evaluation.
        chunks (List[Dict[str, Any]]): List of text chunk dictionaries.
        max_chunks (int): Maximum number of chunks to return.
        min_score (float): Minimum relevance score (0.0-1.0) for chunks to be included.

    Returns:
        List[Dict[str, Any]]: List of top relevant chunks, sorted by relevance score.
    """
    tasks = [check_relevance(question, chunk) for chunk in chunks]
    gathered_chunks = await asyncio.gather(*tasks)
    
    # Filter by relevance and minimum score
    relevant_chunks = [
        chunk for chunk in gathered_chunks 
        if chunk.get('relevance', {}).get('relevance') and 
           chunk.get('relevance_score', 0.0) >= min_score
    ]
    
    # Sort by relevance score (highest first)
    relevant_chunks.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
    
    # Limit to top N chunks
    return relevant_chunks[:max_chunks]


async def query_all_chunks(question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Query each relevant text chunk to obtain an answer to the question.

    This function concurrently processes each chunk by retrieving an answer and
    appending it to the chunk's dictionary.

    Args:
        question (str): The question to be answered.
        chunks (List[Dict[str, Any]]): List of text chunks that passed the relevance filter.

    Returns:
        List[Dict[str, Any]]: List of chunks updated with answers.
    """
    tasks = [get_answer(question, chunk) for chunk in chunks]
    return await asyncio.gather(*tasks)


async def reflect_answers(question: str, chunks: List[Dict[str, Any]], model: str = 'openai/gpt-4o-mini') -> Any:
    """
    Reflect on the answers from different text chunks to derive a consolidated answer.
    If no good answer can be synthesized, returns the no_info_response from question metadata.

    Args:
        question (str): The question to be reflected upon.
        chunks (List[Dict[str, Any]]): List of text chunks with answers.
        model (str, optional): Model identifier for the API call. Defaults to 'openai/gpt-4o-mini'.

    Returns:
        Any: The final consolidated answer parsed from the API response.
    """
    # Get question metadata to access no_info_response
    question_metadata = get_question_metadata(question)
    no_info_response = question_metadata.get('no_info_response', 'Information not found in the provided text.')
    
    formatted_chunks: str = "\n".join(
        f"""
<Reference text chunk_id:{chunk.get('chunk_id', 'N/A')}>
{chunk.get('text', '')}
</Reference text chunk_id:{chunk.get('chunk_id', 'N/A')}>
<Answer chunk_id:{chunk.get('chunk_id', 'N/A')}>
{chunk.get('answer', '')}
</Answer chunk_id:{chunk.get('chunk_id', 'N/A')}>
        """.strip() for chunk in chunks
    )

    prompt: str = f"""
Please reflect on the following answers from different text fragments of a paper. 
Cross-reference between them to determine the correct answer for the question.

<Question>
{question}
</Question>

{formatted_chunks}

IMPORTANT: If the chunks do not contain sufficient or coherent information to answer the question properly, 
or if the information is contradictory, incomplete, or unclear, respond with: "INSUFFICIENT_INFO"

Otherwise, provide a well-synthesized answer based on the available information.
    """.strip()

    messages = [{"content": prompt, "role": "user"}]

    for i in range(RETRY):
        try:
            response = await acompletion(model=model, messages=messages, response_format=AnswerWithChunkId,temperature=0)
            result = json.loads(response.choices[0].message.content)
            logger.info("Reflected answer: %s", result)
            return result
        except Exception as e:
            logger.error("Error reflecting answers: %s", e)
            time.sleep(1)



# --------------------------------------------------------------------------
# Batch Processing Helpers with tqdm progress bar
# --------------------------------------------------------------------------

def get_question_metadata(question_text: str) -> dict:
    """
    Get metadata for a specific question from the YAML configuration.
    
    Args:
        question_text (str): The question text to look up.
        
    Returns:
        dict: Question metadata including instructions, output_format, examples, etc.
    """
    question_lower = question_text.lower()
    
    for question_key, question_config in QUESTIONS_CONFIG.items():
        if question_config.get('question', '').lower() in question_lower or question_key.lower() in question_lower:
            return {
                'question_key': question_key,
                'question': question_config.get('question', ''),
                'instructions': question_config.get('instructions', []),
                'output_format': question_config.get('output_format', ''),
                'example_output': question_config.get('example_output', []),
                'bad_example_output': question_config.get('bad_example_output', []),
                'max_chunks': question_config.get('max_chunks', 5),
                'min_score': question_config.get('min_score', 0.4),
                'no_info_response': question_config.get('no_info_response', 'Information not found in the provided text.'),
                'description': question_config.get('description', 'Default configuration')
            }
    
    return {}

def should_use_no_info_response(question: str, chunks: List[Dict[str, Any]], final_answer: str) -> bool:
    """
    Determine if the no_info_response should be used instead of the current answer.
    
    Args:
        question (str): The question being answered.
        chunks (List[Dict[str, Any]]): List of relevant chunks used.
        final_answer (str): The final synthesized answer.
        
    Returns:
        bool: True if no_info_response should be used, False otherwise.
    """
    # Check for explicit insufficient info response
    if final_answer == 'INSUFFICIENT_INFO':
        return True
    
    # Check for insufficient information indicators in the answer
    insufficient_indicators = [
        'not specified', 'not mentioned', 'not described', 'not reported',
        'unclear', 'ambiguous', 'contradictory', 'incomplete', 'no information',
        'cannot determine', 'insufficient data', 'limited information'
    ]
    
    answer_lower = final_answer.lower()
    if any(indicator in answer_lower for indicator in insufficient_indicators):
        return True
    
    # Check if chunks have very low relevance scores
    if chunks:
        avg_relevance = sum(chunk.get('relevance_score', 0.0) for chunk in chunks) / len(chunks)
        if avg_relevance < 0.4:  # Very low average relevance
            return True
    
    # Check if answer is too generic or vague
    generic_indicators = [
        'the study', 'the research', 'the paper', 'the authors',
        'more research needed', 'further study required', 'additional investigation'
    ]
    
    if any(indicator in answer_lower for indicator in generic_indicators):
        # Only use no_info_response if the answer is very generic
        if len(final_answer.split()) < 20:  # Very short, generic answer
            return True
    
    return False

def assess_answer_quality(question: str, chunks: List[Dict[str, Any]], final_answer: str) -> dict:
    """
    Assess the quality of the final answer based on available chunks and question requirements.
    
    Args:
        question (str): The question being answered.
        chunks (List[Dict[str, Any]]): List of relevant chunks used.
        final_answer (str): The final synthesized answer.
        
    Returns:
        dict: Quality assessment including confidence and recommendations.
    """
    question_metadata = get_question_metadata(question)
    
    # Check if answer contains the expected format/patterns
    output_format = question_metadata.get('output_format', '')
    example_outputs = question_metadata.get('example_output', [])
    
    quality_metrics = {
        'confidence': 'medium',
        'issues': [],
        'recommendations': []
    }
    
    # Check for insufficient information indicators
    insufficient_indicators = [
        'not specified', 'not mentioned', 'not described', 'not reported',
        'unclear', 'unclear', 'ambiguous', 'contradictory', 'incomplete'
    ]
    
    answer_lower = final_answer.lower()
    if any(indicator in answer_lower for indicator in insufficient_indicators):
        quality_metrics['confidence'] = 'low'
        quality_metrics['issues'].append('Answer contains insufficient information indicators')
        quality_metrics['recommendations'].append('Consider using no_info_response')
    
    # Check chunk relevance scores
    if chunks:
        avg_relevance = sum(chunk.get('relevance_score', 0.0) for chunk in chunks) / len(chunks)
        if avg_relevance < 0.5:
            quality_metrics['confidence'] = 'low'
            quality_metrics['issues'].append(f'Low average relevance score: {avg_relevance:.2f}')
            quality_metrics['recommendations'].append('Review relevance thresholds')
    
    # Check if answer matches expected format
    if output_format and not any(keyword in answer_lower for keyword in output_format.lower().split()):
        quality_metrics['confidence'] = 'medium'
        quality_metrics['issues'].append('Answer may not match expected output format')
    
    return quality_metrics

def get_relevance_summary(chunks: List[Dict[str, Any]]) -> dict:
    """
    Get a summary of relevance scores for a list of chunks.
    
    Args:
        chunks (List[Dict[str, Any]]): List of chunks with relevance information.
        
    Returns:
        dict: Summary statistics of relevance scores.
    """
    if not chunks:
        return {"total_chunks": 0, "relevant_chunks": 0, "avg_score": 0.0}
    
    relevant_chunks = [c for c in chunks if c.get('relevance', {}).get('relevance')]
    scores = [c.get('relevance_score', 0.0) for c in relevant_chunks]
    
    return {
        "total_chunks": len(chunks),
        "relevant_chunks": len(relevant_chunks),
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "score_distribution": {
            "high": len([s for s in scores if s >= 0.7]),
            "medium": len([s for s in scores if 0.4 <= s < 0.7]),
            "low": len([s for s in scores if s < 0.4])
        }
    }

def chunked(lst: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split a list into smaller sublists (batches) of a given size.

    Args:
        lst (List[Any]): The list to be split.
        batch_size (int): Maximum size of each sublist.

    Returns:
        List[List[Any]]: List of sublists each with a length up to batch_size.
    """
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


async def process_batches_async(
        question: str,
        chunks: List[Dict[str, Any]],
        batch_size: int,
        process_func: Callable[[str, List[Dict[str, Any]]], Any],
        desc: str = "Processing batches"
) -> List[Any]:
    """
    Process text chunks in asynchronous batches using the provided processing function.

    This function divides the full list of chunks into smaller batches and processes
    each batch concurrently using the specified asynchronous function. A tqdm progress
    bar is displayed to indicate progress over batches.

    Args:
        question (str): The question used in the processing steps.
        chunks (List[Dict[str, Any]]): List of text chunks to be processed.
        batch_size (int): Number of chunks to process concurrently in each batch.
        process_func (Callable): An asynchronous function that processes a batch of chunks.
        desc (str, optional): Description for the progress bar. Defaults to "Processing batches".

    Returns:
        List[Any]: Aggregated results from all batches.
    """
    results: List[Any] = []
    batches = chunked(chunks, batch_size)
    for batch in tqdm(batches, desc=desc, unit="batch"):
        batch_results = await process_func(question, batch)
        results.extend(batch_results)
    return results


# --------------------------------------------------------------------------
# Main Execution Flow
# --------------------------------------------------------------------------

async def ask_json(question: str = None, json_path: str=None, batch_size=256) -> None:
    """
    Main asynchronous entry point for processing text chunks to extract and reflect on answers.

    Steps performed:
      1. Load JSON data containing text chunks.
      2. Filter chunks based on relevance to the question.
      3. Query each relevant chunk to retrieve an answer.
      4. Reflect on the collected answers to generate a final consolidated answer.
    """

    if question is None:
        question: str = "What is the main topic of the paper?"
    if json_path is None:
        json_path: str = "papers/001/pages/main_p01-02.pdf.json"

    # Load JSON data from file.
    json_obj: Dict[str, Any] = load_json_file(json_path)
    chunks: List[Dict[str, Any]] = json_obj.get('data', {}).get('chunks', [])
    BATCH_SIZE: int = batch_size

    # Step 1: Get question-specific configuration
    question_config = get_question_config(question)
    logger.info(f"Question config: {question_config}")
    
    # Step 2: Filter out irrelevant chunks with question-specific settings.
    relevant_chunks: List[Dict[str, Any]] = await process_batches_async(
        question, chunks, BATCH_SIZE, 
        lambda q, c: filter_all_chunks(q, c, question_config['max_chunks'], question_config['min_score']), 
        desc="Filtering chunks"
    )

    if len(relevant_chunks) == 0:
        logger.info("No relevant chunks found for the question: %s", question)
        return {
            "answer": question_config.get('no_info_response', 'Information not found in the provided text.'),
            "chunk_ids": [],
            "reason": f"No relevant chunks found for the question. Min score threshold: {question_config['min_score']}",
            "relevance_info": {
                "total_chunks_processed": len(chunks),
                "relevant_chunks_found": 0,
                "question_config": question_config,
                "chunk_scores": []
            },
            "question_metadata": get_question_metadata(question),
            "quality_assessment": {
                "confidence": "high",
                "issues": ["No relevant chunks found"],
                "recommendations": ["Threshold may be too high"]
            }
        }
    
    # Log relevance scores for debugging
    logger.info(f"Found {len(relevant_chunks)} relevant chunks:")
    for i, chunk in enumerate(relevant_chunks):
        chunk_id = chunk.get('chunk_id', 'N/A')
        score = chunk.get('relevance_score', 0.0)
        logger.info(f"  Chunk {i+1}: ID {chunk_id}, Score {score:.2f}")

    # Step 2: Query each relevant chunk for its answer.
    answered_chunks: List[Dict[str, Any]] = await process_batches_async(
        question, relevant_chunks, BATCH_SIZE, query_all_chunks, desc="Querying chunks"
    )
    # Step 3: Reflect on all collected answers to produce the final answer.
    final_result: Any = await reflect_answers(question, answered_chunks)
    # final_result: Any = await process_batches_async(
    #     question, answered_chunks, BATCH_SIZE, reflect_answers, desc="Reflecting answers"
    # )

    logger.info("Final result: %s", final_result)
    
    # Check if the reflection stage determined insufficient information
    if isinstance(final_result, dict) and final_result.get('answer') == 'INSUFFICIENT_INFO':
        logger.info("Reflection stage determined insufficient information, using no_info_response")
        final_result['answer'] = question_config.get('no_info_response', 'Information not found in the provided text.')
        final_result['reason'] = 'Insufficient or incoherent information found in relevant chunks'
    
    # Get question metadata for enhanced output
    question_metadata = get_question_metadata(question)
    
    # Assess the quality of the final answer
    answer_quality = assess_answer_quality(question, relevant_chunks, final_result.get('answer', ''))
    
    # Ensure the final_result has the required structure
    if isinstance(final_result, dict):
        # Ensure required fields exist
        if 'answer' not in final_result:
            final_result['answer'] = question_config.get('no_info_response', 'Information not found in the provided text.')
        if 'reason' not in final_result:
            final_result['reason'] = 'Answer generated from available information'
        if 'chunk_ids' not in final_result:
            final_result['chunk_ids'] = [chunk.get('chunk_id', 'N/A') for chunk in relevant_chunks]
    else:
        # If final_result is not a dict, create the proper structure
        final_result = {
            'answer': str(final_result),
            'reason': 'Answer generated from available information',
            'chunk_ids': [chunk.get('chunk_id', 'N/A') for chunk in relevant_chunks]
        }
    
    # Create the enhanced result with the required structure
    enhanced_result = {
        "answer": final_result.get('answer', ''),
        "reason": final_result.get('reason', ''),
        "chunk_ids": final_result.get('chunk_ids', []),
        # Additional metadata fields
        "relevance_info": {
            "total_chunks_processed": len(chunks),
            "relevant_chunks_found": len(relevant_chunks),
            "question_config": question_config,
            "chunk_scores": [
                {
                    "chunk_id": chunk.get('chunk_id', 'N/A'),
                    "relevance_score": chunk.get('relevance_score', 0.0),
                    "reason": chunk.get('relevance', {}).get('reason', 'N/A')
                }
                for chunk in relevant_chunks
            ]
        },
        "question_metadata": question_metadata,
        "quality_assessment": answer_quality
    }
    
    pprint(enhanced_result)
    return enhanced_result

# Entry point when running as a script.
if __name__ == "__main__":
    asyncio.run(ask_json())
