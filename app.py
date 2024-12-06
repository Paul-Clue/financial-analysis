from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
import dotenv
import json
import yfinance as yf
import concurrent.futures
from langchain_community.embeddings import HuggingFaceEmbeddings
# from google.colab import userdata
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np
import requests
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

pinecone_api_key = os.getenv('PINECONE_API_KEY')

index_name = "stocks"
namespace = "stock-descriptions"

hf_embeddings = HuggingFaceEmbeddings()
vectorstore = PineconeVectorStore(index_name=index_name, embedding=hf_embeddings)


pc = Pinecone(api_key=pinecone_api_key)

# Connect to your Pinecone index
pinecone_index = pc.Index(index_name)

groq_api_key = os.getenv('GROQ_API_KEY')

client = OpenAI(
  base_url="https://api.groq.com/openai/v1",
  api_key=groq_api_key
)

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# query = "apple"

# system_prompt = f"""You are an expert in the field of embeddings, cosine similarity search and vector databases. Given the following query for a vector database with a pinecone index that stock descriptions are stored in, please provide a better query that will return more relevant results:

# Query: {query}
# """

# llm_response = client.chat.completions.create(
#     model="llama-3.1-70b-versatile",
#     messages=[
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": query}
#     ]
# )

# response = llm_response.choices[0].message.content
# print("RESPONSE", response)


# raw_query_embedding = get_huggingface_embeddings(query)

# top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=10, include_metadata=True, namespace=namespace)

# print("Checking matches for GOOGL:")
# for match in top_matches['matches']:
#     ticker = match['metadata'].get('Ticker')
#     score = match['score']
#     print(f"Ticker: {ticker}, Score: {score}")
#     if ticker == 'GOOGL':
#         print("\nFound GOOGL!")
#         print("Full metadata:", match['metadata'])
# print("top_matches", top_matches)
def HandleQuery(query):
    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=10, include_metadata=True, namespace=namespace)

    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    system_prompt = f"""You are an expert at providing answers about stocks. Please answer my question provided.

    When giving your response, please do not mention the context provided to you or the query.

    Please provide a detailed answer to the question.

    Please provide all of the answers that you receive from the context provided.

    Please provide the answers from most relevant to least relevant.

    Please provide the answer in a markdown format.

    Please be consistent in the markdown format for all of your answers.
    """

    llm_response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    response = llm_response.choices[0].message.content
    return response

# print("RESPONSE", response)
# print("something")

# query = "Which stocks are in the consumer staples sector?"

# Title
st.title('My Stock Analysis App')

# Add a text input for stock symbol
# st.write("### Response:")
query = st.text_input('Ask About Stocks:',)


# Add a button
if st.button('Get Stock Info'):
    st.write(f'Getting info for {query}...')
    response = HandleQuery(query)
    
    # Display the response
    st.write("### Response:")
    st.write(response)
    
    # Optional: Add formatting
    st.markdown("---")  # Add a divider
    
    # You can also add expandable sections
    with st.expander("Show Raw Response"):
        st.code(response)  # Shows response in a code block
        
    # Add error handling
    if not response:
        st.error("No information found for this query.")
    
# Add a sidebar
st.sidebar.title('Options')
show_charts = st.sidebar.checkbox('Show Charts')
