import streamlit as st
import sys
import os

# Add the current directory to the Python path so we can import the modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the individual apps
from chatbot_app_simplified import main as chatbot_main
from visualization_app import main as visualization_main

# Page configuration
st.set_page_config(
    page_title="Policy CoPilot - Comprehensive Analysis Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .tab-container {
        margin-top: 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Policy CoPilot</h1>
        <p>Comprehensive Analysis Platform for Policy Research and Development Outcomes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs([
        "📚 Policy Copilot: Definition & Measurements Analysis", 
        "📊 Policy Copilot: Visualizing and Benchmarking Developmental Outcomes"
    ])
    
    with tab1:
        st.markdown("### 🤖 Policy Copilot ChatBot")
        st.markdown("""
        Welcome to the Policy Copilot ChatBot! This tool helps you analyze policy documents and research 
        literature to understand definitions, measurements, drivers, and variations in developmental outcomes.
        
        **How to use:**
        1. Describe your developmental project
        2. Ask questions about definitions, measurements, drivers, or variations
        3. Get comprehensive answers with citations from our policy document database
        """)
        st.markdown("---")
        
        # Run the chatbot app
        chatbot_main()
    
    with tab2:
        st.markdown("### 📊 Visualization Dashboard")
        st.markdown("""
        Welcome to the Policy CoPilot Visualization Dashboard! This tool provides comprehensive 
        visualizations for analyzing development outcomes across different countries and time periods.
        
        **Available visualizations:**
        - Country-specific time trends
        - Segmented analysis (by gender, age, etc.)
        - Country comparisons
        - Global benchmarking and regional comparisons
        """)
        st.markdown("---")
        
        # Run the visualization app
        visualization_main()

if __name__ == "__main__":
    main()
