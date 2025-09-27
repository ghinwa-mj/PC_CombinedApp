import os
import streamlit as st

# Try to get from Streamlit secrets first, then environment variables
def get_secret(key):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)

WEAVIATE_API_KEY = get_secret('WEAVIATE_API_KEY')
RESTENDPOINT = get_secret('RESTENDPOINT')
OPENAI_API_KEY = get_secret('OPENAI_API_KEY')
ANTHROPIC_API_KEY = get_secret('ANTHROPIC_API_KEY')
GEMINI_API_KEY = get_secret('GEMINI_API_KEY')
