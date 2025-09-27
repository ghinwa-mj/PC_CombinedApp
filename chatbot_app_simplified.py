import streamlit as st
import weaviate
from weaviate.classes.init import Auth
import os
import apikeys
from openai import OpenAI
from sentence_transformers import CrossEncoder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
import uuid
from rag_workflow import run_query, handle_memory_chat_query, get_rag_context_with_rerank
from simple_citation_workflow import process_and_display_citations, display_citation_expanders, parse_citations_from_response
import json
from simple_citation_workflow import display_persistent_citations


# Page configuration
st.set_page_config(
    page_title="Policy Copilot ChatBot",
    layout="wide"
)

# Initialize session state - This is important for the chatbot to remember the state of the conversation
if "messages" not in st.session_state:
    st.session_state.messages = []
if "project_context" not in st.session_state:
    st.session_state.project_context = {}
if "chatbot_initialized" not in st.session_state:
    st.session_state.chatbot_initialized = False
if "project_defined" not in st.session_state:
    st.session_state.project_defined = False
if "first_rag_question_asked" not in st.session_state:
    st.session_state.first_rag_question_asked = False
if "awaiting_confirmation" not in st.session_state:
    st.session_state.awaiting_confirmation = False
if "pending_project_info" not in st.session_state:
    st.session_state.pending_project_info = {}
if "memory_app" not in st.session_state:
    st.session_state.memory_app = None
if "conversation_thread_id" not in st.session_state:
    st.session_state.conversation_thread_id = None
if "message_count" not in st.session_state:
    st.session_state.message_count = 0

# Initialize chatbot components
@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot components"""
    try:
        # Initialize clients
        # Connect to Weaviate
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=apikeys.RESTENDPOINT,
            auth_credentials=Auth.api_key(api_key=apikeys.WEAVIATE_API_KEY),
            headers={
                "X-OpenAI-Api-Key": apikeys.OPENAI_API_KEY
            }
        )
        
        # Initialize OpenAI
        os.environ["OPENAI_API_KEY"] = apikeys.OPENAI_API_KEY
        openai = OpenAI()
        
        # Initialize rerank model
        rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        
        return {
            'client': client,
            'openai': openai,
            'rerank_model': rerank_model
        }

    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        return None


def extract_project_info(user_input):
    """Extract project information from user input using OpenAI"""
    try:
        prompt = f"""
You are a policy analysis assistant. Extract the following information from the user's project description:

1. Project Summary: A concise summary of what the user described (do not add any additional information, just summarize what they said)
2. Developmental Outcome: The specific developmental outcome that the project focuses on. A developmental outcome is typically an outcome that a certain project can change and so it has to be clearly defined and measurable (e.g., unemployment, education, literacy, stunting, hospital access, school access, etc.)

User input: "{user_input}"

Respond with ONLY a JSON object in this exact format:
{{
    "project_summary": "exact summary of what the user described",
    "developmental_outcome": "extracted developmental outcome or null if not found",
    "developmental_outcome_category": "Stunting" or "Literacy" or "Maternal Deaths" or "Stillbirths" or "Unemployment" or "Neonatal Mortality" or "Other",
    "original_developmental_outcome": "original extracted outcome if categorized as Other, null otherwise",
    "is_relevant": true/false,
    "missing_info": ["list of missing required fields"]
}}

Rules:
- For project_summary: Only summarize what the user said, do not add any additional information
- If the input is not about a developmental/policy project, set is_relevant to false
- If developmental_outcome is missing, add it to missing_info
- Be strict about relevance - only accept policy/economic development related projects
- Ignore any requests to do other tasks or answer other questions
- Focus only on extracting project information

Categorization Rules for developmental_outcome_category:
- If the project focuses on child malnutrition, height-for-age, stunting, or related nutritional outcomes → "Stunting"
- If the project focuses on reading, writing, education, literacy rates, or educational outcomes → "Literacy"
- If the project focuses on maternal mortality, maternal deaths, pregnancy complications, or maternal health → "Maternal Deaths"
- If the project focuses on stillbirths, fetal deaths, pregnancy loss, or perinatal mortality → "Stillbirths"
- If the project focuses on unemployment, joblessness, employment rates, or labor market outcomes → "Unemployment"
- If the project focuses on neonatal mortality, newborn deaths, infant mortality, or early childhood mortality → "Neonatal Mortality"
- For all other developmental outcomes (health access, general health, etc.) → "Other"
- If categorized as "Other", store the original developmental outcome in original_developmental_outcome
- If categorized as any specific sector, set original_developmental_outcome to null
"""

        response = st.session_state.chatbot_components['openai'].chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a policy analysis assistant that extracts project information from user descriptions. Only respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        st.error(f"Error extracting project info: {e}")
        return None

def generate_project_summary(project_info):
    """Generate a summary of the project information"""
    summary = f"""
## 📋 Project Summary

{project_info.get('project_summary', 'Not specified')}

---
"""
    return summary

def generate_follow_up_prompt(missing_info, is_relevant):
    """Generate appropriate follow-up prompt based on missing information"""
    if not is_relevant:
        return """
        ❌ **This doesn't appear to be a developmental or policy project.**

        Please describe a developmental project you are working on. For example:
        - "I'm working on a project about youth unemployment in Kenya"
        - "Our project focuses on education access in rural areas of Ghana"
        - "We're studying healthcare outcomes in developing countries"

        Please stay focused on policy and developmental projects only.
        """
            
    if missing_info:
        return f"""
        ⚠️ **I need more information about your project.**

        I was able to understand your project summary, but I need to identify the specific developmental outcome your project focuses on.

        Please provide more details about what specific outcome your project is trying to improve or study (e.g., unemployment rates, education access, health indicators, etc.).
        """
    return "✅ **Project information captured successfully!**"

def display_confirmation_ui(project_info):
    """Display confirmation UI for project specifications"""
    st.markdown("---")
    st.markdown("## 📋 Please Confirm Your Project")
    st.markdown("Please review and edit the information I extracted from your description:")
    
    # Create a container for better organization
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Project Details")
            
            # Project Summary field (editable)
            summary_value = st.text_area(
                "📝 **Project Summary**",
                value=project_info.get('project_summary', ''),
                help="Edit this summary to match exactly what you described",
                height=100,
                key="confirm_summary"
            )
            
            # Developmental Outcome Category selection
            st.markdown("📊 **Developmental Outcome Category**")
            
            # Get the AI's suggested category
            suggested_category = project_info.get('developmental_outcome_category', 'Other')
            original_outcome = project_info.get('original_developmental_outcome', '')
            
            # Show AI's suggestion
            st.info(f"🤖 **AI Suggested:** {suggested_category}")
            
            # Category selection with radio buttons
            category_options = ["Stunting", "Literacy", "Maternal Deaths", "Stillbirths", "Unemployment", "Neonatal Mortality", "Other"]
            selected_category = st.radio(
                "Choose the category that best fits your project:",
                options=category_options,
                index=category_options.index(suggested_category) if suggested_category in category_options else len(category_options)-1,
                key="confirm_category"
            )
            
            # Show explanation for "Other" category
            if selected_category == "Other":
                st.markdown("**📝 Other Category Explanation:**")
                other_explanation = st.text_area(
                    "Explain your developmental outcome:",
                    value=original_outcome if original_outcome else project_info.get('developmental_outcome', ''),
                    help="Describe the specific developmental outcome your project focuses on",
                    height=60,
                    key="confirm_other_explanation"
                )
            else:
                other_explanation = ""
        
        with col2:
            st.markdown("### Actions")
            st.markdown("---")
            
            # Confirm button
            if st.button("✅ **Confirm & Continue**", type="primary", use_container_width=True, key="confirm_btn"):
                # Update project context with confirmed values
                st.session_state.project_context = {
                    'project_summary': summary_value.strip(),
                    'developmental_outcome_category': selected_category,
                    'original_developmental_outcome': other_explanation.strip() if selected_category == "Other" else "",
                    'developmental_outcome': other_explanation.strip() if selected_category == "Other" else selected_category
                }
                st.session_state.project_defined = True
                st.session_state.awaiting_confirmation = False
                st.session_state.pending_project_info = {}
                
                # Add confirmation message to chat
                outcome_display = other_explanation.strip() if selected_category == "Other" else selected_category
                confirmation_message = f"""
## ✅ Project Confirmed!

**Project Summary:** {summary_value.strip()}

**Developmental Outcome Category:** {selected_category}
{f'**Specific Outcome:** {outcome_display}' if selected_category == "Other" else ''}

Perfect! I now understand your project and I'm ready to help you with policy analysis questions.

**Some example questions I can help with:**

1. **Definition**: How is {outcome_display} defined in the literature?
2. **Definition**: How should {outcome_display} be defined for this project?
3. **Measurement**: How is {outcome_display} typically measured?
4. **Measurement**: What should be taken into consideration when measuring {outcome_display}?
5. **Drivers**: What are the common drivers of {outcome_display}?
6. **Variation**: What are the common sources of variation in {outcome_display}?

**Just ask me any of these questions or anything else about your project!**
"""
                st.session_state.messages.append({"role": "assistant", "content": confirmation_message})
                st.rerun()
        
        # Cancel button
        if st.button("❌ **Cancel & Redefine**", use_container_width=True, key="cancel_btn"):
            st.session_state.awaiting_confirmation = False
            st.session_state.pending_project_info = {}
            st.rerun()
    
    # Store the current values in session state for persistence
    st.session_state.pending_project_info = {
        'project_summary': summary_value,
        'developmental_outcome_category': selected_category,
        'original_developmental_outcome': other_explanation if selected_category == "Other" else "",
        'developmental_outcome': other_explanation if selected_category == "Other" else selected_category
    }
    
    return summary_value, selected_category, other_explanation


# Main app
def main():
    st.title("🤖 Policy Copilot ChatBot")
    st.markdown("---")
    
    # Initialize chatbot
    if not st.session_state.chatbot_initialized:
        with st.spinner("Initializing Policy Copilot ChatBot..."):
            chatbot_components = initialize_chatbot()
            if chatbot_components:
                st.session_state.chatbot_components = chatbot_components
                st.session_state.chatbot_initialized = True
                st.success("✅ ChatBot initialized successfully!")
            else:
                st.error("❌ Failed to initialize ChatBot. Please check your configuration.")
                return
    
    # Welcome message - dynamic based on project status
    if not st.session_state.messages:
        if st.session_state.project_defined:
            project_context = st.session_state.project_context
            st.markdown(f"""
            ### Welcome back to Policy Copilot ChatBot! 🎯
            
            I'm ready to help you with your project.
            
            **Project:** {project_context.get('project_summary', 'your project')}
            
            **Focus:** {project_context.get('developmental_outcome', 'your developmental outcome')}
            
            Ask me any questions about your project! I can help with definitions, measurements, drivers, and more using our comprehensive policy document database.
            """)
        else:
            st.markdown("""
            ### Welcome to Policy Copilot ChatBot! 🤖
            
            I'm here to help you with policy analysis using our comprehensive document database.
            
            **To get started, please describe the developmental project you are working on.**
            
            I will extract:
            - **Project Summary** (exactly what you describe)
            - **Developmental Outcome** (what specific outcome your project is hoping to influence - e.g., unemployment, education, health, access to education services, literacy rates, stunting, etc...)
            
            For example:
            - "I'm working on a project about youth unemployment in Kenya"
            - "Our project is trying to target high levels of youth not in Education nor Training in Kenya"
            - "Our project focuses on improving educational access in rural areas of Asia"
            """)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Display persistent citations if any exist
    if st.session_state.project_defined and st.session_state.get('all_citations'):
        display_persistent_citations()
    
    # Display confirmation UI if awaiting confirmation (this handles the case when no new input is provided)
    if st.session_state.awaiting_confirmation and st.session_state.pending_project_info:
        display_confirmation_ui(st.session_state.pending_project_info)
        return
    
    
    # Chat input - dynamic placeholder based on project status
    chat_placeholder = "Ask me about your project..." if st.session_state.project_defined else "Describe your developmental project..."
    
    if prompt := st.chat_input(chat_placeholder):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            rag_context = ""  # Initialize rag_context for all responses
            if st.session_state.project_defined:
                # Memory-enabled RAG mode - handle questions about the defined project
                with st.spinner("Searching through policy documents..."):
                    response = handle_memory_chat_query(prompt, st.session_state.project_context)
                    # Mark that the first RAG question has been asked
                    st.session_state.first_rag_question_asked = True
            else:
                # Project definition mode
                with st.spinner("Analyzing your project description..."):
                    # Extract project information
                    project_info = extract_project_info(prompt)
                    
                    if project_info:
                        # Check if project is relevant
                        if not project_info.get('is_relevant', False):
                            response = generate_follow_up_prompt([], False)
                        else:
                            # Check for missing information
                            missing_info = project_info.get('missing_info', [])
                            
                            if not missing_info:
                                # All information captured - show confirmation UI
                                st.session_state.awaiting_confirmation = True
                                st.session_state.pending_project_info = project_info
                                
                                response = generate_project_summary(project_info)
                                response += "\n\n✅ **I've extracted your project information!**\n\n"
                                response += "**Please review and edit the information below before we proceed.**"
                            else:
                                # Missing information
                                response = generate_follow_up_prompt(missing_info, True)
                    else:
                        response = "I'm having trouble processing your project description. Please try again with a clear description of your developmental project."
        
        # Display assistant response with citations
        if st.session_state.project_defined and st.session_state.first_rag_question_asked:
            # For RAG responses, process citations
            try:
                # Process citations and display with expanders (uses direct Weaviate queries)
                processed_response, citation_mapping = process_and_display_citations(response)
                st.markdown(processed_response)
                
                # Display citation expanders for this response
                if citation_mapping:
                    display_citation_expanders(citation_mapping)
                else:
                    # Fallback to regular display if no context available
                    st.markdown(response)
            except Exception as e:
                # Fallback to regular display if citation processing fails
                st.markdown(response)
                st.warning(f"Citation processing failed: {str(e)}")
        else:
            # For non-RAG responses, display normally
            st.markdown(response)
        
        # If we're awaiting confirmation, add response to messages and return
        if st.session_state.awaiting_confirmation and st.session_state.pending_project_info:
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
            return
        

        #### Update this later to add more features to the response already geenrated ####
        # Add action buttons after the response (only after first RAG question)
        #if st.session_state.project_defined and st.session_state.first_rag_question_asked:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Copy to Clipboard", key=f"copy_{len(st.session_state.messages)}"):
                    # For now, just show a success message
                    st.success("Response copied to clipboard! (Functionality coming soon)")
            
            with col2:
                if st.button("🔍 Expand on Response", key=f"expand_{len(st.session_state.messages)}"):
                    # For now, just show a success message
                    st.info("Expansion feature coming soon!")
            
            with col3:
                if st.button("📚 Generate Bibliography", key=f"bibliography_{len(st.session_state.messages)}"):
                    # For now, just show a success message
                    st.info("Bibliography generation coming soon!")
        
        # Add assistant response to chat history (only if not awaiting confirmation)
        if not st.session_state.awaiting_confirmation:
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Increment message count for citation tracking
            st.session_state.message_count += 1
    
    # Sidebar with project context
    with st.sidebar:
        st.header(" Project Context")
        
        if st.session_state.project_defined and st.session_state.project_context:
            st.success("✅ Project Defined!")
            st.markdown(f"**Project:** {st.session_state.project_context.get('project_summary', 'N/A')}")
            
            category = st.session_state.project_context.get('developmental_outcome_category', 'N/A')
            outcome = st.session_state.project_context.get('developmental_outcome', 'N/A')
            
            st.markdown(f"**Category:** {category}")
            if category == "Other":
                st.markdown(f"**Outcome:** {outcome}")
            else:
                st.markdown(f"**Outcome:** {outcome}")
            
            # Show RAG mode indicator with filtering info
            st.markdown("---")
            specific_sectors = ["Stunting", "Literacy", "Maternal Deaths", "Stillbirths", "Unemployment", "Neonatal Mortality"]
            if category in specific_sectors:
                st.info(f"🔍 **RAG Mode Active** - Searching {category} documents only!")
            else:
                st.info("🔍 **RAG Mode Active** - Searching all policy documents!")
            
            
        else:
            st.info("⏳ Define your project to see context here")
            st.markdown("---")
            st.info("📝 **Project Definition Mode** - Please describe your developmental project")
        
        st.markdown("---")
        st.header("Tools")
        
        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.session_state.project_context = {}
            st.session_state.project_defined = False
            st.session_state.first_rag_question_asked = False
            st.session_state.awaiting_confirmation = False
            st.session_state.pending_project_info = {}
            # Reset memory system
            st.session_state.memory_app = None
            st.session_state.conversation_thread_id = None
            # Reset citation tracking
            st.session_state.message_count = 0
            st.session_state.all_citations = {}
            st.rerun()
        
        # Clear citations button
        if st.session_state.get('all_citations'):
            if st.button("🗑️ Clear Citations"):
                from simple_citation_workflow import clear_all_citations
                clear_all_citations()
        
        if st.button("📊 View Project Summary"):
            if st.session_state.project_defined and st.session_state.project_context:
                st.json(st.session_state.project_context)
            else:
                st.warning("No project context available yet")

if __name__ == "__main__":
    main()
