# database.py

import pandas as pd
import numpy as np
import openai
from redis import Redis
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.query import Query

# Configuration constants
from config import EMBEDDINGS_MODEL, PREFIX, VECTOR_FIELD_NAME, INDEX_NAME

# Setting OpenAI API key from the configuration file.
import config
openai.api_key = config.DevelopmentConfig.OPENAI_KEY

# Get a connection to a Redis server
def get_redis_connection(host='localhost', port=6379, db=0):
    r = Redis(host=host, port=port, db=db, decode_responses=False)
    return r

# Load vector embeddings and associated metadata to Redis using a pipeline.
# Pipelining batches multiple commands into a single step for faster processing.
def load_vectors(client: Redis, input_list, vector_field_name):
    p = client.pipeline(transaction=False)
    for text in input_list:
        key = f"{PREFIX}:{text['id']}"
        item_metadata = text['metadata']
        item_keywords_vector = np.array(text['vector'], dtype='float32').tobytes()
        item_metadata[vector_field_name] = item_keywords_vector
        p.hset(key, mapping=item_metadata)
    p.execute()

# Query Redis for the closest matches to the embedded user query.
# This function uses the KNN (k-nearest neighbors) algorithm to find the most similar embedded vectors.
# database.py

def query_redis(redis_conn, query, index_name, top_k=5):  # Default top_k set to 5
    """Query Redis using KNN."""
    embedded_query = np.array(openai.Embedding.create(
        input=query,
        model=EMBEDDINGS_MODEL,
    )["data"][0]['embedding'], dtype=np.float32).tobytes()
    
    q = Query(f'*=>[KNN {top_k} @{VECTOR_FIELD_NAME} $vec_param AS vector_score]')
    q = q.sort_by('vector_score').paging(0, top_k).return_fields('vector_score', 'filename', 'text_chunk', 'file_chunk_index').dialect(2)
    params_dict = {"vec_param": embedded_query}
    results = redis_conn.ft(index_name).search(q, query_params=params_dict)
    return results

def get_redis_results(redis_conn, query, index_name, top_k=5):
    """Process Redis search results into a DataFrame."""
    query_result = query_redis(redis_conn, query, index_name, top_k=top_k)
    query_result_list = []
    for i, result in enumerate(query_result.docs):
        result_order = i
        text = result.text_chunk
        score = result.vector_score
        query_result_list.append((result_order, text, score))
    result_df = pd.DataFrame(query_result_list, columns=['id', 'result', 'certainty'])
    return result_df

