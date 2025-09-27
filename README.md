# Policy CoPilot - Combined Analysis Platform

This directory contains a comprehensive Streamlit application that combines two powerful tools for policy analysis and developmental outcomes research.

## Features

### Tab 1: Policy Copilot: Definition & Measurements Analysis
- **ChatBot Interface**: Interactive AI-powered chatbot for policy document analysis
- **Project Definition**: Define your developmental project and outcomes
- **RAG System**: Retrieval-Augmented Generation using Weaviate vector database
- **Citation Management**: Automatic citation extraction and display
- **Memory System**: Conversation memory for contextual responses

### Tab 2: Policy Copilot: Visualizing and Benchmarking Developmental Outcomes
- **Interactive Visualizations**: Multiple chart types for data analysis
- **Country Comparisons**: Compare developmental outcomes across countries
- **Time Trends**: Analyze trends over time for specific countries
- **Segmented Analysis**: Break down data by gender, age, and other dimensions
- **Global Benchmarking**: Compare countries against global and regional averages

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run main_app.py
   ```

3. **Access the App**:
   - Open your browser and go to `http://localhost:8501`
   - Use the tabs at the top to switch between the two main features

## File Structure

- `main_app.py` - Main Streamlit application with tabbed interface
- `chatbot_app_simplified.py` - ChatBot functionality (Tab 1)
- `visualization_app.py` - Visualization dashboard (Tab 2)
- `rag_workflow.py` - RAG system implementation for the chatbot
- `simple_citation_workflow.py` - Citation management system
- `apikeys.py` - API keys configuration
- `requirements.txt` - Python dependencies

## Prerequisites

- Python 3.8 or higher
- Valid API keys for:
  - OpenAI (for LLM functionality)
  - Weaviate (for vector database)
  - Access to the Weaviate cloud instance

## Usage Tips

1. **For ChatBot (Tab 1)**:
   - Start by describing your developmental project
   - Ask specific questions about definitions, measurements, drivers, or variations
   - Use the citation expanders to view source documents

2. **For Visualizations (Tab 2)**:
   - Select your development outcome indicator
   - Choose the type of analysis you want to perform
   - Use the country selection tools to focus on specific regions

## Support

For issues or questions, please refer to the individual component documentation or contact the development team.
