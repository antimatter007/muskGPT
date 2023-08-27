# Required imports
import openai
import os
import numpy as np
import pandas as pd
import tiktoken
import textract
from database import get_redis_connection, get_redis_results
from transformers import handle_file_string

# Constants and default settings
from config import COMPLETIONS_MODEL, EMBEDDINGS_MODEL, CHAT_MODEL, TEXT_EMBEDDING_CHUNK_SIZE, VECTOR_FIELD_NAME, PREFIX, INDEX_NAME

# Set up the OpenAI API key from the config file
import config
openai.api_key = config.DevelopmentConfig.OPENAI_KEY

# Redis specific imports
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

# Setting up Redis client
redis_client = get_redis_connection()

# Other constants
VECTOR_DIM = 1536
DISTANCE_METRIC = "COSINE"
location = 'data'
query = ''

# Get all PDF files from the specified directory
def getPDFFiles():
    data_dir = os.path.join(os.curdir, location)
    pdf_files = sorted([x for x in os.listdir(data_dir) if 'DS_Store' not in x])
    return pdf_files, data_dir

# Create a database index if it doesn't exist
def createDatabaseIndex():
    # Define RediSearch fields to store dataset columns
    filename = TextField("filename")
    text_chunk = TextField("text_chunk")
    file_chunk_index = NumericField("file_chunk_index")
    # RediSearch vector fields for embeddings
    text_embedding = VectorField(VECTOR_FIELD_NAME,
        "HNSW", {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": DISTANCE_METRIC
        }
    )
    fields = [filename, text_chunk, file_chunk_index, text_embedding]
    
    try:
        # Check if index already exists
        redis_client.ft(INDEX_NAME).info()
        print(f"Index {INDEX_NAME} already exists")
    except Exception as e:
        # If not, create a new index
        redis_client.ft(INDEX_NAME).create_index(fields=fields, 
            definition=IndexDefinition(prefix=[PREFIX], 
            index_type=IndexType.HASH))
        print(f"Index {INDEX_NAME} was created succesfully")

    return True

# Add PDF documents to the Redis index
def addDocumentsToIndex():
    pdf_files, data_dir = getPDFFiles()
    # Initialize a tokenizer for text processing
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        # Extract raw text from the PDF
        text = textract.process(pdf_path, method='pdfminer')
        # Chunk the document, embed the content and load it into Redis
        handle_file_string((pdf_file, text.decode("utf-8")), tokenizer, redis_client, VECTOR_FIELD_NAME, INDEX_NAME)

# Query the Redis database
def queryRedisDatabase():
    result_df = get_redis_results(redis_client, query, index_name=INDEX_NAME)
    redis_result = result_df['result'][0]
    
    messages = [{"role": "system", "content": "Your name is Karabo. You are a helpful assistant."}]
    
    ENGINEERING_PROMPT = f"""
    Answer this question: {query}
    Attempt to answer the question based on this content: {redis_result}
    """
    messages.append({'role': 'user', 'content': ENGINEERING_PROMPT})

    response = openai.ChatCompletion.create(model=CHAT_MODEL, messages=messages)
    
    try:
        answer = response['choices'][0]['message']['content'].replace('\n', '<br>')
    except:
        answer = 'Oops you beat the AI, try a different question, if the problem persists, come back later.'

    return answer

# Custom function to get an answer from GPT based on a query
def customChatGPTAnswer(the_query):
    result_df = get_redis_results(redis_client, the_query, index_name=INDEX_NAME)
    redis_result = result_df['result'][0]
    
    messages = [{"role": "system", "content": "Your name is Karabo. You are a helpful assistant."}]
    
    ENGINEERING_PROMPT = f"""
    Answer this question: {the_query}
    Attempt to answer the question based on this content: {redis_result}
    """
    messages.append({'role': 'user', 'content': ENGINEERING_PROMPT})

    response = openai.ChatCompletion.create(model=CHAT_MODEL, messages=messages)
    
    try:
        answer = response['choices'][0]['message']['content'].replace('\n', '<br>')
    except:
        answer = 'Oops you beat the AI, try a different question, if the problem persists, come back later.'

    return answer
