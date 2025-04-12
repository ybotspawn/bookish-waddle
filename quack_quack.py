import numpy as np
import os
import tiktoken 
from openai import OpenAI
import openai
import tiktoken
import pandas as pd
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def build_source_stigs(source_stig_csv):
    df = pd.read_csv(source_stig_csv)    
    num_tokens_from_string(df['summary'][0], "cl100k_base")
    df['token_count'] = df['summary'].apply(lambda text:num_tokens_from_string(text, "cl100k_base"))
    df['token_count'].sum() * .0004/1000 # One time cost
    df['embedding'] = df['summary'].apply(get_embedding)
    df.to_csv('stig_embeddings.csv')
    return df

def get_source_stigs(source_stig_csv="stig_embeddings.csv"):
    df = pd.read_csv(source_stig_csv)
    return df

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

def get_source_vid(stig_title, df):
    prompt = stig_title # "Directory Browsing on the IIS 10.0 website must be disabled"
    prompt_embedding = get_embedding(prompt)
    df['prompt_similarity'] = df['embedding'].apply(lambda vector: vector_similarity(vector, prompt_embedding))
    return df.nlargest(1, 'prompt_similarity').iloc[0]['v-id']

def vector_similarity(vector1, vector2):
    # Compare the embedding vector with the prompt vector and rank it based on similarity
    # The prompt similarity does not need to be written out to the data file so that we can reuse the data file
    return np.dot(np.array(vector1), np.array(vector2))

# Decorator here to validate the stig_embeddings.csv exists
def crossref_stigs(target_stig_file):
    # have to pass each IIS 10.0 stig in to the promp embedding and df.nlargest function
    targetstig = pd.read_csv(target_stig_file)
    targetstig['legacy_id'] = targetstig['summary'].apply(get_source_vid) # effectively uses panda to do our for each over each value and applies it to our new field legacy_id
    targetstig.to_csv('stig_combined.csv')

