import numpy as np
import argparse
import os
import tiktoken 
from openai import OpenAI
import openai
import pandas as pd
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

SOURCE_EMBEDDING_FILE = "stig_embeddings.csv"

class FilePathException(Exception):
    def __init__(self, filepath):
        self.message = f"Filepath: {filepath} either not provided or does not exist"
    def __str__(self):
        return self.message
    
def build_source_embeddings(source_stig_csv="stigs.csv"):
    source_dataframe = pd.read_csv(source_stig_csv)    
    num_tokens_from_string(source_dataframe['summary'][0], "cl100k_base")
    source_dataframe['token_count'] = source_dataframe['summary'].apply(lambda text:num_tokens_from_string(text, "cl100k_base"))
    source_dataframe['token_count'].sum() * .0004/1000 # One time cost
    source_dataframe['embedding'] = source_dataframe['summary'].apply(get_embedding)
    source_dataframe.to_csv('stig_embeddings.csv')
    return source_dataframe

def get_source_stig_embeddings(source_stig_csv="stig_embeddings.csv"):
    source_dataframe = pd.read_csv(source_stig_csv)
    return source_dataframe

def num_tokens_from_string(string, encoding_name):
    # cl100k base is the one for text embedding
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# This is what costs money, then you save and can reuse the data file over and over
def get_embedding(text):
    result = client.embeddings.create(
        model='text-embedding-ada-002', ### Ada domensions to 1536 dimensions
        input=text
    )
    return result.data[0].embedding


def check_embedding_file(func):
    def wrapper():
        if (source_embeddings and os.path.exists(source_embeddings)):
            pass
        elif (os.path.exists(SOURCE_EMBEDDING_FILE)):
            source_embeddings = SOURCE_EMBEDDING_FILE
        elif (not source_embeddings and not os.path.exists(SOURCE_EMBEDDING_FILE)): # if the source_embeddings is not provided and the default location does not exist, throw an error
            raise(FilePathException(SOURCE_EMBEDDING_FILE))

def get_source_vid(stig_title, source_dataframe):
    prompt_embedding = get_embedding(stig_title) # "Directory Browsing on the IIS 10.0 website must be disabled"
    #source_dataframe = get_source_stig_embeddings(source_embeddings)
    #source_dataframe = build_source_embeddings()
    source_dataframe['prompt_similarity'] = source_dataframe['embedding'].apply(lambda vector: vector_similarity(vector, prompt_embedding))
    if bool(source_dataframe.nlargest(1, 'prompt_similarity').iloc[0]['prompt_similarity'] < .9):
        return "None"
    else:
        return source_dataframe.nlargest(1, 'prompt_similarity').iloc[0]['v-id']

def vector_similarity(vector1, vector2):
    # Compare the embedding vector with the prompt vector and rank it based on similarity
    # The prompt similarity does not need to be written out to the data file so that we can reuse the data file
    return np.dot(np.array(vector1), np.array(vector2))

# Decorator here to validate an embeddings file exists in the current directory or that a path has been provided
@check_embedding_file
def crossref_stigs(target_stig_file):
    targetstig = pd.read_csv(target_stig_file)
    source_embeddings = build_source_embeddings() # testing purposes get_source_stig_embeddings()
    targetstig['legacy-id'] = targetstig['summary'].apply(get_source_vid, args=(source_embeddings)) # effectively uses panda to do our for each over each value and applies it to our new field legacy_id
    targetstig.to_csv('stig_combined.csv')

#variable name for embedding file will be source_embeddings; this will need to be set in our argparse