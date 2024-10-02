# text_transformers.py

import logging
import openai
import numpy as np
import pandas as pd
import os
import tiktoken
import fitz  # PyMuPDF
from typing import Iterator
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from database import get_redis_connection, load_vectors, get_redis_results
import config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("text_transformers.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set OpenAI API Key from Configuration
openai.api_key = config.DevelopmentConfig.OPENAI_KEY

# Initialize Redis client
redis_client = get_redis_connection()

# Constants
VECTOR_DIM = 1536
DISTANCE_METRIC = "COSINE"
PREFIX = "asp"
INDEX_NAME = "asp-index"
CHAT_MODEL = 'gpt-3.5-turbo'
EMBEDDINGS_MODEL = "text-embedding-ada-002"
VECTOR_FIELD_NAME = "content_vector"
TEXT_EMBEDDING_CHUNK_SIZE = 300  # Adjust as needed
location = 'data'  # Directory containing PDF files

def queryRedisDatabase(user_query):
    """Query the Redis database and get a GPT-based answer."""
    logger.info(f"Received query: {user_query}")

    # Retrieve results from Redis
    result_df = get_redis_results(redis_client, user_query, index_name=INDEX_NAME, top_k=5)  # Increased top_k to 5
    logger.info(f"Number of results retrieved from Redis: {len(result_df)}")

    if result_df.empty:
        logger.info("No relevant information found for the query.")
        return "No relevant information found."

    # Aggregate all relevant text chunks
    redis_results = result_df['result'].tolist()
    aggregated_text = "\n".join(redis_results)
    logger.debug(f"Full aggregated text: {aggregated_text}")
    logger.info(f"Aggregated text from Redis: {aggregated_text[:500]}...")  # Log first 500 chars for brevity

    # Construct the prompt
    messages = [
        {"role": "system", "content": "Your name is Karabo. You are a helpful assistant."},
        {"role": "user", "content": (
            f"Answer the following question based on the provided information:\n\n"
            f"Question: {user_query}\n\n"
            f"Information:\n{aggregated_text}\n\n"
            f"Answer:"
        )}
    ]

    logger.debug(f"Prompt sent to OpenAI: {messages}")

    try:
        response = openai.ChatCompletion.create(model=CHAT_MODEL, messages=messages)
        answer = response['choices'][0]['message']['content'].replace('\n', '<br>')
        logger.info("Received response from OpenAI successfully.")
    except Exception as e:
        logger.error(f"Error generating response from OpenAI: {e}")
        answer = 'Sorry, I encountered an error while generating your response.'

    return answer

def get_col_average_from_list_of_lists(list_of_lists):
    """Calculate the average of each column in a list of lists."""
    if len(list_of_lists) == 1:
        return list_of_lists[0]
    else:
        list_of_lists_array = np.array(list_of_lists)
        average_embedding = np.average(list_of_lists_array, axis=0)
        return average_embedding.tolist()

def create_embeddings_for_text(text, tokenizer):
    """Create embeddings for a given text using a tokenizer."""
    token_chunks = list(chunks(text, TEXT_EMBEDDING_CHUNK_SIZE, tokenizer))
    text_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]

    embeddings_response = get_embeddings(text_chunks, EMBEDDINGS_MODEL)
    embeddings = [embedding["embedding"] for embedding in embeddings_response]
    text_embeddings = list(zip(text_chunks, embeddings))

    average_embedding = get_col_average_from_list_of_lists(embeddings)

    return (text_embeddings, average_embedding)

def get_embeddings(text_array, model):
    """Fetch embeddings from OpenAI API."""
    logger.info(f"Requesting embeddings for {len(text_array)} chunks.")
    response = openai.Embedding.create(input=text_array, model=model)
    logger.info("Embeddings received successfully.")
    return response["data"]

def chunks(text, n, tokenizer):
    """Split text into smaller chunks."""
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
    """Process a file string and upload its embeddings to a Redis database."""
    filename = file[0]
    file_body_string = file[1]

    # Clean up the text
    clean_file_body_string = file_body_string.replace("  ", " ").replace("\n", "; ").replace(';', ' ')

    text_to_embed = f"Filename is: {filename}; {clean_file_body_string}"

    logger.info(f"Creating embeddings for file: {filename}")
    # Generate embeddings
    try:
        text_embeddings, average_embedding = create_embeddings_for_text(text_to_embed, tokenizer)
        logger.info(f"Generated {len(text_embeddings)} embeddings for {filename}.")
    except Exception as e:
        logger.error(f"Error creating embedding for {filename}: {e}")
        return  # Exit the function if embedding creation fails

    # Prepare vectors for Redis upload
    vectors = []
    for i, (text_chunk, embedding) in enumerate(text_embeddings):
        id = f"{filename}-!{i}"
        vectors.append({
            'id': id,
            'vector': embedding,
            'metadata': {
                "filename": filename,
                "text_chunk": text_chunk,
                "file_chunk_index": i
            }
        })

    logger.info(f"Uploading {len(vectors)} vectors to Redis for {filename}.")
    # Upload to Redis
    try:
        load_vectors(redis_conn, vectors, text_embedding_field)
        logger.info(f"Uploaded embeddings for {filename} successfully.")
    except Exception as e:
        logger.error(f"Ran into a problem uploading to Redis for {filename}: {e}")

def createDatabaseIndex():
    """Create a new index in the Redis database."""
    # Define fields for indexing
    filename = TextField("filename")
    text_chunk = TextField("text_chunk")
    file_chunk_index = NumericField("file_chunk_index")
    text_embedding = VectorField(
        VECTOR_FIELD_NAME, "HNSW",
        {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": DISTANCE_METRIC,
            "INITIAL_CAP": 1000,
            "M": 16,
            "EF_CONSTRUCTION": 200,
        }
    )
    fields = [filename, text_chunk, file_chunk_index, text_embedding]

    logger.info(f"Attempting to create index {INDEX_NAME}.")

    # Create a new index if it doesn't exist
    try:
        redis_client.ft(INDEX_NAME).info()
        logger.info(f"Index {INDEX_NAME} already exists.")
    except Exception as e:
        logger.warning(f"Index {INDEX_NAME} does not exist. Attempting to create it.")
        try:
            redis_client.ft(INDEX_NAME).create_index(
                fields=fields,
                definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
            )
            logger.info(f"Index {INDEX_NAME} was created successfully.")
        except Exception as create_e:
            logger.error(f"Failed to create index {INDEX_NAME}: {create_e}")

def getPDFFiles():
    """Retrieve all PDF files from the data directory."""
    data_dir = os.path.join(os.curdir, location)
    pdf_files = [x for x in os.listdir(data_dir) if x.lower().endswith('.pdf') and 'DS_Store' not in x]
    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}.")
    return sorted(pdf_files), data_dir

def addDocumentsToIndex():
    """Read and index PDF documents using PyMuPDF."""
    pdf_files, data_dir = getPDFFiles()
    tokenizer = tiktoken.get_encoding("cl100k_base")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        try:
            logger.info(f"Processing file: {pdf_file}")
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            handle_file_string((pdf_file, text), tokenizer, redis_client, VECTOR_FIELD_NAME, INDEX_NAME)
            logger.info(f"Indexed {pdf_file} successfully.")
        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {e}")

# Main execution block
if __name__ == '__main__':
    logger.info("Starting database index creation.")
    createDatabaseIndex()
    logger.info("Adding documents to the index.")
    addDocumentsToIndex()
    logger.info("Completed indexing process.")
