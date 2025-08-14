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
    """
    reason: str
    relevance: bool
    is_bib_list: bool


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
    """.strip()

    messages = [{"content": prompt, "role": "user"}]

    for i in range(RETRY):
        try:
            response = await acompletion(model=model, messages=messages, response_format=Relevance,temperature=0)
            outcome = json.loads(response.choices[0].message.content)

            chunk['relevance'] = outcome

            if outcome.get('is_bib_list'):
                logger.info("Detected bibliography list for chunk %s", chunk.get('chunk_id'))
                # Mark the chunk as irrelevant if it is a bibliography list.
                chunk['relevance'] = {'relevance': False, 'reason': 'Detected as bibliography list.'}

            logger.info("Relevance for chunk %s: %s", chunk.get('chunk_id'), outcome)
            break
        except Exception as e:
            logger.error("Error checking relevance for chunk %s: %s", chunk.get('chunk_id'), e)
            # Mark the chunk as irrelevant in case of any error.
            chunk['relevance'] = {'relevance': False, 'reason': str(e)}
            time.sleep(1)
            continue
    return chunk


async def filter_all_chunks(question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter a list of text chunks to retain only those relevant to the question.

    This function concurrently checks the relevance of each chunk and filters out
    those that are not relevant.

    Args:
        question (str): The question used for relevance evaluation.
        chunks (List[Dict[str, Any]]): List of text chunk dictionaries.

    Returns:
        List[Dict[str, Any]]: List of chunks deemed relevant.
    """
    tasks = [check_relevance(question, chunk) for chunk in chunks]
    gathered_chunks = await asyncio.gather(*tasks)
    return [chunk for chunk in gathered_chunks if chunk.get('relevance', {}).get('relevance')]


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

    Constructs a detailed prompt that includes the text and answer from each chunk,
    then calls the API to produce a final answer that cross-references all inputs.

    Args:
        question (str): The question to be reflected upon.
        chunks (List[Dict[str, Any]]): List of text chunks with answers.
        model (str, optional): Model identifier for the API call. Defaults to 'openai/gpt-4o'.

    Returns:
        Any: The final consolidated answer parsed from the API response.
    """
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

    # Step 1: Filter out irrelevant chunks.
    relevant_chunks: List[Dict[str, Any]] = await process_batches_async(
        question, chunks, BATCH_SIZE, filter_all_chunks, desc="Filtering chunks"
    )

    if len(relevant_chunks) == 0:
        logger.info("No relevant chunks found for the question: %s", question)
        return {
            "answer": "Information not found in the provided text.",
            "chunk_ids": [],
            "reason": "No relevant chunks found for the question."
        }

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
    pprint({
        "question": question,
        "result": final_result
    })

    return final_result

# Entry point when running as a script.
if __name__ == "__main__":
    asyncio.run(ask_json())
