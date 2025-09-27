import os
import streamlit as st

# Try to get from Streamlit secrets first, then environment variables, then fallback values
def get_secret(key, fallback=None):
    try:
        # Try Streamlit secrets first
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except:
        pass
    
    # Try environment variables
    env_value = os.getenv(key)
    if env_value:
        return env_value
    
    # Return fallback if provided
    return fallback

# API Keys - replace fallback values with your actual keys for local development
# For local development, replace the placeholder values below with your actual API keys
WEAVIATE_API_KEY = get_secret('WEAVIATE_API_KEY', 'your_weaviate_api_key_here')
RESTENDPOINT = get_secret('RESTENDPOINT', 'your_weaviate_endpoint_here')
OPENAI_API_KEY = get_secret('OPENAI_API_KEY', 'your_openai_api_key_here')
ANTHROPIC_API_KEY = get_secret('ANTHROPIC_API_KEY', 'your_anthropic_api_key_here')
GEMINI_API_KEY = get_secret('GEMINI_API_KEY', 'your_gemini_api_key_here')
