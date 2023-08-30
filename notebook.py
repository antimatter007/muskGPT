# Required Libraries
import openai
import os
import numpy as np
import pandas as pd
import tiktoken
import textract

# Local Module Imports
from database import get_redis_connection, get_redis_results
from transformers import handle_file_string

# Configuration
from config import COMPLETIONS_MODEL, EMBEDDINGS_MODEL, CHAT_MODEL, TEXT_EMBEDDING_CHUNK_SIZE, VECTOR_FIELD_NAME, PREFIX, INDEX_NAME
import config

# Set OpenAI API Key from Configuration
openai.api_key = config.DevelopmentConfig.OPENAI_KEY

# Redis Imports for Database and Search
from redis import Redis
from redis.commands.search.query import Query
from redis.commands.search.field import (
    TextField,
    VectorField,
    NumericField
)
from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)

# Initialize Redis client
redis_client = get_redis_connection()

# Constants
VECTOR_DIM = 1536
DISTANCE_METRIC = "COSINE"
location = 'data'
query = ''

# Function to get PDF files from a directory
def getPDFFiles():
    data_dir = os.path.join(os.curdir, location)
    pdf_files = [x for x in os.listdir(data_dir) if 'DS_Store' not in x]
    return sorted(pdf_files), data_dir

# Function to create a new index in the Redis database
def createDatabaseIndex():
    # Define fields for indexing
    filename = TextField("filename")
    text_chunk = TextField("text_chunk")
    file_chunk_index = NumericField("file_chunk_index")
    text_embedding = VectorField(
        VECTOR_FIELD_NAME, "HNSW",
        {"TYPE": "FLOAT32", "DIM": VECTOR_DIM, "DISTANCE_METRIC": DISTANCE_METRIC}
    )
    fields = [filename, text_chunk, file_chunk_index, text_embedding]

    # Create a new index if it doesn't exist
    try:
        redis_client.ft(INDEX_NAME).info()
        print(f"Index {INDEX_NAME} already exists")
    except Exception:
        redis_client.ft(INDEX_NAME).create_index(
            fields=fields,
            definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
        )
        print(f"Index {INDEX_NAME} was created successfully")

# Function to read and index PDF documents
def addDocumentsToIndex():
    pdf_files, data_dir = getPDFFiles()
    tokenizer = tiktoken.get_encoding("cl100k_base")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        text = textract.process(pdf_path, method='pdfminer')
        handle_file_string((pdf_file, text.decode("utf-8")), tokenizer, redis_client, VECTOR_FIELD_NAME, INDEX_NAME)

# Function to query the Redis database and get a GPT-based answer
def queryRedisDatabase():
    result_df = get_redis_results(redis_client, query, index_name=INDEX_NAME)
    redis_result = result_df['result'][0]
    messages = [{"role": "system", "content": "Your name is Karabo. You are a helpful assistant."}]
    ENGINEERING_PROMPT = f"Answer this question: {query}\nAttempt to answer the question based on this content: {redis_result}"
    messages.append({'role': 'user', 'content': ENGINEERING_PROMPT})
    response = openai.ChatCompletion.create(model=CHAT_MODEL, messages=messages)

    try:
        answer = response['choices'][0]['message']['content'].replace('\n', '<br>')
    except:
        answer = 'Oops you beat the AI, try a different question, if the problem persists, come back later.'

    return answer
