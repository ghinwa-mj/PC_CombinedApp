import streamlit as st
import weaviate
from weaviate.classes.init import Auth
import os
import apikeys
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
import uuid
import tiktoken


def estimate_tokens(text):
    """Estimate token count for text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4


def get_optimal_limits(context_text, max_tokens=100000):
    """Dynamically adjust limits based on context size"""
    current_tokens = estimate_tokens(context_text)
    
    if current_tokens > max_tokens:
        # Reduce limits if context is too large
        reduction_factor = max_tokens / current_tokens
        return max(50, int(200 * reduction_factor)), max(30, int(150 * reduction_factor))
    
    return 200, 150


##Basic RAG Workflow
def run_query(query, num_docs=300, sector_filter=None):
    """Run RAG query using Weaviate and OpenAI with reranking"""
    try:
        # Get the chatbot components from session state
        chatbot_components = st.session_state.get('chatbot_components', {})
        client = chatbot_components.get('client')
        openai = chatbot_components.get('openai')
        rerank_model = chatbot_components.get('rerank_model')
        
        if not all([client, openai, rerank_model]):
            return "Error: Chatbot components not properly initialized."
        
        # Use the existing get_rag_context_with_rerank function
        context = get_rag_context_with_rerank(query, initial_limit=200, sector_filter=sector_filter)
        
        if context == "No context available - RAG system not properly initialized.":
            return context
        if context == "No relevant documents found.":
            return context
        if context.startswith("Error retrieving context:"):
            return context

        # Create system prompt based on project context
        project_context = st.session_state.get('project_context', {})
        system_prompt = create_system_prompt(project_context)
        
        # Create user prompt using the centralized function
        prompt = create_user_prompt(query, context, project_context)

        # Generate response using OpenAI
        answer = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return answer.choices[0].message.content
        
    except Exception as e:
        return f"Error processing your question: {str(e)}"



##Run Workflow with Memory
def handle_memory_chat_query(user_question, project_context):
    """Handle chat queries using memory-enabled RAG system"""
    try:
        # Initialize memory app if not already done
        if not st.session_state.memory_app:
            chatbot_components = st.session_state.get('chatbot_components', {})
            openai_client = chatbot_components.get('openai')
            if openai_client:
                st.session_state.memory_app = create_memory_workflow(openai_client, project_context)
                # Generate a unique thread ID for this conversation
                st.session_state.conversation_thread_id = str(uuid.uuid4())
            else:
                return "Error: OpenAI client not available for memory system."
        
        # Convert Streamlit messages to LangChain messages
        langchain_messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        # Add the new user message
        langchain_messages.append(HumanMessage(content=user_question))
        
        # Create the state for LangGraph
        state = {"messages": langchain_messages}
        
        # Run the memory-enabled workflow
        config = {"configurable": {"thread_id": st.session_state.conversation_thread_id}}
        result = st.session_state.memory_app.invoke(state, config=config)
        
        # Extract the response
        if result and "messages" in result:
            # Get the last message (should be the AI response)
            last_message = result["messages"][-1]
            
            if hasattr(last_message, 'content'):
                return last_message.content
            else:
                return str(last_message)
        else:
            # Fallback to direct RAG response if memory system fails
            # Determine sector filter based on project context
            category = project_context.get('developmental_outcome_category', 'Other')
            specific_sectors = ['Stunting', 'Literacy', 'Maternal Deaths', 'Stillbirths', 'Unemployment', 'Neonatal Mortality']
            sector_filter = category if category in specific_sectors else None
            rag_result = run_query(user_question, num_docs=300, sector_filter=sector_filter)
            return rag_result
            
    except Exception as e:
        # Fallback to direct RAG response if memory system fails
        try:
            # Determine sector filter based on project context
            category = project_context.get('developmental_outcome_category', 'Other')
            specific_sectors = ['Stunting', 'Literacy', 'Maternal Deaths', 'Stillbirths', 'Unemployment', 'Neonatal Mortality']
            sector_filter = category if category in specific_sectors else None
            rag_result = run_query(user_question, num_docs=300, sector_filter=sector_filter)
            return rag_result
        except Exception as rag_error:
            return f"I encountered an error while processing your question: {str(e)}. RAG fallback also failed: {str(rag_error)}"


def get_rag_context_with_rerank(user_question, initial_limit=200, final_limit=150, sector_filter=None):
    """Get RAG context with reranking for better relevance"""
    try:
        # Get chatbot components for RAG
        chatbot_components = st.session_state.get('chatbot_components', {})
        if not chatbot_components:
            return "No context available - RAG system not properly initialized."
        
        # Generate query vector using OpenAI API
        openai = chatbot_components.get('openai')
        client = chatbot_components.get('client')
        rerank_model = chatbot_components.get('rerank_model')
        
        if not all([openai, client, rerank_model]):
            return "No context available - RAG system not properly initialized."
        
        # Use OpenAI text-embedding-3-large API
        embedding_response = openai.embeddings.create(
            model="text-embedding-3-large",
            input=user_question
        )
        query_vector = embedding_response.data[0].embedding
        
        # Query Weaviate with larger limit for reranking
        # Note: We'll do post-filtering since near_vector doesn't support where clause directly
        docs = client.collections.get("Feature1").query.near_vector(
            near_vector=query_vector,
            limit=initial_limit,
            return_properties=["in_text_citation", "document_title", "content", "page_number", "chunk_index", "sector"]
        )
        # Prepare documents for reranking
        documents = []
        for obj in docs.objects:
            doc = {
                'object': obj,
                'title': obj.properties.get('document_title', 'Unknown Document'),
                'content': obj.properties.get('content', ''),
                'citation': obj.properties.get('in_text_citation', 'No citation available'),
                'page': obj.properties.get('page_number', 'Unknown page'),
                'chunk_index': obj.properties.get('chunk_index', 'Unknown chunk'),
                'sector': obj.properties.get('sector', 'Unknown sector')
            }
            documents.append(doc)
        
        # Apply sector filtering if specified (post-filtering)
        if sector_filter:
            filtered_documents = []
            for doc in documents:
                doc_sector = doc['sector'].lower() if doc['sector'] else ''
                if sector_filter.lower() in doc_sector:
                    filtered_documents.append(doc)
            
            # Debug output
            st.info(f"🔍 **Debug Info:** Found {len(documents)} total documents, {len(filtered_documents)} match sector '{sector_filter}'")
            
            documents = filtered_documents
            
            # If no documents match the sector filter, return early
            if not documents:
                return f"No documents found for sector '{sector_filter}'. Please try a different question or check if documents exist for this sector."
        
        if len(documents) == 0:
            return "No relevant documents found."
        
        # Create query-document pairs for reranking
        query_doc_pairs = []
        for doc in documents:
            # Combine title and content for reranking
            doc_text = f"{doc['title']} {doc['content']}"
            query_doc_pairs.append([user_question, doc_text])
        
        # Get reranking scores
        rerank_scores = rerank_model.predict(query_doc_pairs)
        
        # Add rerank scores to documents
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(rerank_scores[i])
        
        # Sort by rerank score (higher is better)
        documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Take top final_limit results
        top_documents = documents[:final_limit]
        
        # Create context with token monitoring
        context_parts = []
        available_chunks = []  # Debug: track available chunk indices
        current_context_size = 0
        max_context_tokens = 100000  # Conservative limit to stay under 128k
        
        for doc in top_documents:
            obj = doc['object']
            doc_title = obj.properties.get('document_title', 'Unknown Document')
            in_text_citation = obj.properties.get('in_text_citation', 'No citation available')
            content = obj.properties.get('content', '')
            page_number = obj.properties.get('page_number', 'Unknown page')
            chunk_index = obj.properties.get('chunk_index', 'Unknown chunk')
            
            doc_section = f"\nDOCUMENT: {doc_title}\nCITATION: {in_text_citation}\nCHUNK: {chunk_index}\nPAGE: {page_number}\nCONTENT:\n{content}\n"
            
            # Check if adding this document would exceed token limit
            doc_tokens = estimate_tokens(doc_section)
            if current_context_size + doc_tokens > max_context_tokens:
                break
                
            context_parts.append(doc_section)
            available_chunks.append(chunk_index)  # Debug: collect chunk indices
            current_context_size += doc_tokens
        
        
        final_context = "\n\n".join(context_parts)
        
        # Debug output for context creation
        if sector_filter:
            st.info(f"📚 **Context Created:** {len(context_parts)} document chunks, {current_context_size} tokens, sector filter: '{sector_filter}'")
        else:
            st.info(f"📚 **Context Created:** {len(context_parts)} document chunks, {current_context_size} tokens, no sector filter")
        
        return final_context
        
    except Exception as e:
        return f"Error retrieving context: {str(e)}"

def create_system_prompt(project_context):
    """Create system prompt based on project context"""
    country = project_context.get('country', 'your country')
    developmental_outcome = project_context.get('developmental_outcome', 'your developmental outcome')
    
    return f'''You are a Policy Advisor working as a Thought Partner on a project in {country}, focusing on {developmental_outcome}. 
    You will be responsible for analyzing large amounts of policy and academic documents.
    You will not provide policy recommendations, but act as a partner helping the user make effective analysis and have a clear understanding of the policy landscape. 
    
    Rules:
    - Rely on the information provided (The Literature) to respond accurately to the user question. 
    - Use neutral language.
    - Do not indicate bias or opinion. 
    - Use language such as "Research finds", "the literature suggests", etc.
    - Always cite your sources with in-text citations.
    - Avoid implying causal relationships unless explicitly supported by the literature.
    - The provided chat history includes the last two conversations (Question & Answer pairs) to maintain context.'''

def create_user_prompt(query, context, project_context):
    """Create user prompt with context and rules"""
    country = project_context.get('country', 'your country')
    
    return f"""Following your system prompt and the context below, answer the question faithfully. 

        Rules:
        
        - Ensure that the response is as comprehensive as possible within the context of {country} or within the geographical scope mentioned by the user
        - If no country specific information is available, start with information on regions similar to {country} and the world.
        - Avoid implying factors are causal when they are not. Simply explain the relationship between the factors and the outcome based on the literature.
        - Even when prompted by the user, do not provide any form of advice or policy recommendations - Whether personal or on the project level. Simply explain the information based on the literature.
        - IMPORTANT: You MUST cite your sources. For every piece of information you provide, include a citation in this EXACT format: (Author, Year) [Chunk Number]
        - Use ONLY the chunk numbers provided in the context. Each document chunk has a CHUNK field showing its number.
        - For multiple citations, use: (Author, Year) [Chunk Number] (Author, Year) [Chunk Number]
        - Do NOT use any other citation format. Do NOT use "Chunk X" in parentheses. Use ONLY the format specified above.
        - If there is no information available in the literature, do not make up information. Simply state that the information is not available in the literature and prompt the user to change the question to a topic that is more suitable for the literature.
        
        Context:
        {context}

        Question: {query}
        """

def call_model_with_rag(state: MessagesState, model, project_context):
    """Enhanced model call that integrates RAG with memory"""
    # Get the latest user message
    latest_message = state["messages"][-1]
    user_question = latest_message.content
    
    # Get RAG context with reranking
    # Determine sector filter based on project context
    project_context = st.session_state.get('project_context', {})
    category = project_context.get('developmental_outcome_category', 'Other')
    specific_sectors = ['Stunting', 'Literacy', 'Maternal Deaths', 'Stillbirths', 'Unemployment', 'Neonatal Mortality']
    sector_filter = category if category in specific_sectors else None
    
    context = get_rag_context_with_rerank(user_question, initial_limit=200, final_limit=150, sector_filter=sector_filter)
    
    # Store the RAG context in session state for citation processing
    st.session_state.last_rag_context = context
    
    # Rest of the function remains the same...
    # Create system prompt
    system_prompt = create_system_prompt(project_context)
    system_message = SystemMessage(content=system_prompt)
    message_history = state["messages"][:-1]  # exclude the most recent user input
    
    # Create the full prompt with context using the centralized function
    full_prompt = create_user_prompt(user_question, context, project_context)
    
    # Use last two conversations (4 messages: 2 Q&A pairs) for memory
    if len(message_history) >= 4:
        # Get the last 4 messages (2 Q&A pairs)
        recent_messages = message_history[-4:]
        human_message = HumanMessage(content=full_prompt)
        message_updates = model.invoke([system_message] + recent_messages + [human_message])
    else:
        # Use all available message history
        human_message = HumanMessage(content=full_prompt)
        message_updates = model.invoke([system_message] + message_history + [human_message])

    return {"messages": message_updates}

def create_memory_workflow(openai_client, project_context):
    """Create LangGraph workflow with memory for the chatbot"""
    try:
        # Create LangChain ChatOpenAI model
        model = ChatOpenAI(
            model="gpt-4.1",
            temperature=0,
            openai_api_key=apikeys.OPENAI_API_KEY
        )
        
        # Create workflow
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_node("model", lambda state: call_model_with_rag(state, model, project_context))
        workflow.add_edge(START, "model")
        
        # Add memory checkpointing
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        
        return app
        
    except Exception as e:
        st.error(f"Failed to create memory workflow: {e}")
        return None