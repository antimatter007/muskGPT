from typing import Iterator
from numpy import array, average
import openai
import pandas as pd
import numpy as np

from config import TEXT_EMBEDDING_CHUNK_SIZE, EMBEDDINGS_MODEL
from database import load_vectors

# Set OpenAI API key from the configuration file
import config
openai.api_key = config.DevelopmentConfig.OPENAI_KEY

def get_col_average_from_list_of_lists(list_of_lists):
    """Calculate the average of each column in a list of lists.
    
    Args:
        list_of_lists: A list of lists where each inner list is a column.
        
    Returns:
        A list containing the average of each column.
    """
    if len(list_of_lists) == 1:
        return list_of_lists[0]
    else:
        list_of_lists_array = array(list_of_lists)
        average_embedding = average(list_of_lists_array, axis=0)
        return average_embedding.tolist()

def create_embeddings_for_text(text, tokenizer):
    """Create embeddings for a given text using a tokenizer.
    
    Args:
        text: The text to be embedded.
        tokenizer: The tokenizer used for text preprocessing.
        
    Returns:
        A tuple containing:
        - List of tuples (text_chunk, embedding) for each chunk of the text.
        - An average embedding for the entire text.
    """
    token_chunks = list(chunks(text, TEXT_EMBEDDING_CHUNK_SIZE, tokenizer))
    text_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]

    embeddings_response = get_embeddings(text_chunks, EMBEDDINGS_MODEL)
    embeddings = [embedding["embedding"] for embedding in embeddings_response]
    text_embeddings = list(zip(text_chunks, embeddings))

    average_embedding = get_col_average_from_list_of_lists(embeddings)

    return (text_embeddings, average_embedding)

def get_embeddings(text_array, engine):
    """Fetch embeddings from OpenAI engine.
    
    Args:
        text_array: List of text chunks to be embedded.
        engine: The OpenAI engine used for embedding.
        
    Returns:
        A list of embeddings.
    """
    return openai.Engine(id=engine).embeddings(input=text_array)["data"]

def chunks(text, n, tokenizer):
    """Split text into smaller chunks.
    
    Args:
        text: The text to be chunked.
        n: The approximate size for each chunk.
        tokenizer: The tokenizer used for text preprocessing.
        
    Yields:
        Chunks of tokens from the text.
    """
    tokens = tokenizer.encode(text)
    i = 0
    while i < len(tokens):
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j

def handle_file_string(file, tokenizer, redis_conn, text_embedding_field, index_name):
    """Process a file string and upload its embeddings to a Redis database.
    
    Args:
        file: A tuple containing filename and file content as a string.
        tokenizer: The tokenizer used for text preprocessing.
        redis_conn: Redis connection object.
        text_embedding_field: The Redis field where embeddings are stored.
        index_name: The index name for the Redis database.
    """
    filename = file[0]
    file_body_string = file[1]

    # Clean up the text
    clean_file_body_string = file_body_string.replace("  ", " ").replace("\n", "; ").replace(';', ' ')

    text_to_embed = "Filename is: {}; {}".format(filename, clean_file_body_string)

    # Generate embeddings
    try:
        text_embeddings, average_embedding = create_embeddings_for_text(text_to_embed, tokenizer)
    except Exception as e:
        print(f"Error creating embedding: {e}")

    # Prepare vectors for Redis upload
    vectors = []
    for i, (text_chunk, embedding) in enumerate(text_embeddings):
        id = f"{filename}-!{i}"
        vectors.append({'id': id, 'vector': embedding, 'metadata': {"filename": filename, "text_chunk": text_chunk, "file_chunk_index": i}})

    # Upload to Redis
    try:
        load_vectors(redis_conn, vectors, text_embedding_field)
    except Exception as e:
        print(f"Ran into a problem uploading to Redis: {e}")

class BatchGenerator:
    """Generate batches from a DataFrame."""
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size

    def to_batches(self, df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Split DataFrame into smaller batches.
        
        Args:
            df: The DataFrame to be split.
            
        Yields:
            Smaller DataFrames as batches.
        """
        splits = self.splits_num(df.shape[0])
        if splits <= 1:
            yield df
        else:
            for chunk in np.array_split(df, splits):
                yield chunk

    def splits_num(self, elements: int) -> int:
        """Calculate the number of splits needed for batching.
        
        Args:
            elements: The total number of elements in the DataFrame.
            
        Returns:
            The number of splits needed.
        """
        return round(elements / self.batch_size)

    __call__ = to_batches
