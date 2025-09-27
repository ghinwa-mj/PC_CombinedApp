import streamlit as st
import apikeys

st.title("API Keys Test")

st.write("**RESTENDPOINT:**", apikeys.RESTENDPOINT)
st.write("**WEAVIATE_API_KEY:**", apikeys.WEAVIATE_API_KEY[:20] + "..." if apikeys.WEAVIATE_API_KEY else "None")
st.write("**OPENAI_API_KEY:**", apikeys.OPENAI_API_KEY[:20] + "..." if apikeys.OPENAI_API_KEY else "None")

if apikeys.RESTENDPOINT and apikeys.RESTENDPOINT != "your_weaviate_endpoint_here":
    st.success("✅ Secrets are working correctly!")
else:
    st.error("❌ Secrets are not being read properly")
