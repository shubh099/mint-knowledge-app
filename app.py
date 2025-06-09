# app.py -- FINAL ROBUST VERSION for Local & Cloud

import streamlit as st
import os

# Used to load the .env file for local development
from dotenv import load_dotenv

from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# --- 1. Load API Keys Robustly ---

# This function checks if we're running on Streamlit Cloud
def is_running_on_streamlit_cloud():
    """Returns True if the app is running on Streamlit Cloud, False otherwise."""
    return "STREAMLIT_SERVER_PORT" in os.environ

# Load .env file only when running locally
if not is_running_on_streamlit_cloud():
    load_dotenv()

# Load keys based on the environment
if is_running_on_streamlit_cloud():
    # When deployed, use Streamlit's secrets manager
    PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = st.secrets.get("PINECONE_ENVIRONMENT")
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
else:
    # When running locally, use environment variables loaded from .env
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

PINECONE_INDEX_NAME = "mint-e-papers" 

# Error checking for keys
if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, GOOGLE_API_KEY]):
    st.error("API keys are not configured. Please add them to your .env file (for local) or Streamlit's secrets manager (for deployment).", icon="🚨")
    st.stop()

# --- Initialize Models and Pinecone Connection ---
try:
    llm = GoogleGenAI(model_name="models/gemini-pro", api_key=GOOGLE_API_KEY, temperature=0.1)
    embed_model = GoogleGenAIEmbedding(model_name="models/embedding-001", api_key=GOOGLE_API_KEY)
    
    Settings.llm = llm
    Settings.embed_model = embed_model

    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    st.error(f"Failed to initialize models or services. Error: {e}", icon="🚨")
    st.stop()

# --- 2. Load the Knowledge Base and create the Query Engine ---
@st.cache_resource(show_spinner="Connecting to Knowledge Base...")
def get_query_engine():
    """Load the index from Pinecone and create a query engine."""
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    query_engine = index.as_query_engine(similarity_top_k=20)
    
    return query_engine

# --- 3. Build the Streamlit User Interface ---
st.set_page_config(page_title="Mint E-Paper Analysis Engine", layout="centered")
st.title("📄 Mint E-Paper Analysis Engine")
st.markdown("Ask any question about your indexed Mint e-papers. The system will retrieve relevant text and chart data to provide an answer.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! What would you like to know?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

try:
    query_engine = get_query_engine()
except Exception as e:
    st.error(f"Could not connect to the query engine. Error: {e}", icon="🚨")
    st.stop()

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = query_engine.query(prompt)
                st.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
            except Exception as e:
                st.error(f"An error occurred while querying: {e}", icon="🚨")
