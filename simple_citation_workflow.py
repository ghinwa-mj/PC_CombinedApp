import re
import streamlit as st
from weaviate.collections.classes.filters import Filter

def parse_citations_from_response(response_text):
    """
    Parse citations from response text and extract chunk information.
    Handles formats: (Author, Year) [Chunk Number] or (Author, Year, Chunk Index)
    Returns: list of citation dictionaries with author, year, chunk_info
    """
    citations = []
    
    # Pattern 1: (Author, Year) [Chunk Number] - the expected format
    pattern1 = r'\(([^,]+),\s*(\d{4})\)\s*\[(\d+)\]'
    citations_found1 = re.findall(pattern1, response_text)
    
    for author, year, chunk_number in citations_found1:
        citations.append({
            'author': author.strip(),
            'year': year.strip(),
            'chunk_info': chunk_number.strip(),
            'original': f"({author}, {year}) [{chunk_number}]"
        })
    
    # Pattern 2: (Author, Year, Chunk Info) - legacy format for backward compatibility
    pattern2 = r'\(([^,]+),\s*(\d{4}),\s*([^)]+)\)'
    citations_found2 = re.findall(pattern2, response_text)
    
    for author, year, chunk_info in citations_found2:
        citations.append({
            'author': author.strip(),
            'year': year.strip(),
            'chunk_info': chunk_info.strip(),
            'original': f"({author}, {year}, {chunk_info})"
        })
    
    return citations

def extract_chunk_numbers(citations):
    """
    Extract chunk numbers from citation chunk_info.
    Handles both numeric (123) and text (Chunk 4) formats.
    Returns: list of chunk numbers
    """
    chunk_numbers = []
    
    for citation in citations:
        chunk_info = citation['chunk_info']
        
        # Try to extract number from "Chunk X" format
        chunk_match = re.search(r'Chunk\s+(\d+)', chunk_info)
        if chunk_match:
            chunk_numbers.append(int(chunk_match.group(1)))
        else:
            # Try to extract pure number
            try:
                chunk_num = int(chunk_info)
                chunk_numbers.append(chunk_num)
            except ValueError:
                # If it's not a number, skip this citation
                st.warning(f"Could not parse chunk number from: {chunk_info}")
                continue
    
    return chunk_numbers

def get_chunk_data_from_weaviate(chunk_numbers):
    """
    Get chunk data directly from Weaviate using chunk indices.
    direct 1:1 lookup.
    Returns: dictionary mapping chunk_number to chunk_data
    """
    chunk_data = {}
    
    if not chunk_numbers:
        return chunk_data
    
    # Get chatbot components
    chatbot_components = st.session_state.get('chatbot_components', {})
    client = chatbot_components.get('client')
    
    if not client:
        st.warning("Weaviate client not available")
        return chunk_data
    
    
    try:        
        # Query Weaviate for each chunk index
        for chunk_num in chunk_numbers:
            # Query Weaviate for the specific chunk index
            docs = client.collections.get("Feature1").query.fetch_objects(
                filters=Filter.by_property('chunk_index').equal(chunk_num),
                return_properties=["in_text_citation", "document_title", "content", "bibliography", "source", "page_number", "chunk_index", "sector"]
            )
            
            if docs.objects:
                obj = docs.objects[0]  # Should be only one match
                chunk_data[chunk_num] = {
                    'content': obj.properties.get('content', ''),
                    'page_number': obj.properties.get('page_number', 'Unknown page'),
                    'document_title': obj.properties.get('document_title', 'Unknown Document'),
                    'bibliography': obj.properties.get('bibliography', 'No bibliography available'),
                    'in_text_citation': obj.properties.get('in_text_citation', 'No citation available'),
                    'chunk_index': obj.properties.get('chunk_index', 'Unknown chunk'),
                    'source': obj.properties.get('source', 'Unknown Source'),
                    'sector': obj.properties.get('sector', 'Unknown sector')
                }
            else:
                st.warning(f"Chunk {chunk_num} not found in Weaviate")
                
    except Exception as e:
        st.error(f"Error querying Weaviate for chunks: {str(e)}")
    
    return chunk_data

def create_citation_mapping(citations, chunk_data):
    """
    Create a mapping from chunk numbers to citation data.
    Returns: dictionary with chunk_number as key, citation_data as value
    """
    citation_mapping = {}
    
    for citation in citations:
        chunk_info = citation['chunk_info']
        
        # Extract chunk number - handle both formats
        chunk_match = re.search(r'Chunk\s+(\d+)', chunk_info)
        if chunk_match:
            chunk_num = int(chunk_match.group(1))
        else:
            try:
                chunk_num = int(chunk_info)
            except ValueError:
                continue
        
        if chunk_num in chunk_data:
            citation_mapping[chunk_num] = {
                'author': citation['author'],
                'year': citation['year'],
                'chunk_data': chunk_data[chunk_num],
                'original_citation': citation['original']
            }
    
    return citation_mapping

def render_citations_with_expanders(response_text, citation_mapping):
    """
    Render response text with clickable citations using original chunk numbers.
    """
    if not citation_mapping:
        return response_text
    
    # Replace citations with numbered format
    processed_text = response_text
    
    for chunk_num, citation in citation_mapping.items():
        original_citation = citation['original_citation']
        author_year = f"({citation['author']}, {citation['year']})"
        numbered_citation = f"{author_year} [{chunk_num}]"
        
        processed_text = processed_text.replace(original_citation, numbered_citation)
    
    return processed_text

def display_citation_expanders(citation_mapping):
    """
    Display citation expanders below the response text.
    """
    if not citation_mapping:
        return
    
    st.markdown("---")
    st.markdown("**📚 Citations:**")
    
    # Create columns for citations (max 3 per row)
    citations_per_row = 3
    citation_items = list(citation_mapping.items())
    
    for i in range(0, len(citation_items), citations_per_row):
        cols = st.columns(citations_per_row)
        
        for j, (chunk_num, citation_data) in enumerate(citation_items[i:i + citations_per_row]):
            with cols[j]:
                chunk_data = citation_data['chunk_data']
                
                with st.expander(f"📖 [{chunk_num}] {chunk_data['in_text_citation']}", expanded=False):
                    st.markdown(f"**Document:** {chunk_data['document_title']}")
                    st.markdown(f"**Page:** {chunk_data['page_number']}")
                    st.markdown(f"**Bibliography:** {chunk_data['bibliography']}")
                    st.markdown("---")
                    st.markdown("**Content:**")
                    st.markdown(chunk_data['content'])

def process_and_display_citations(response_text, rag_context=None):
    """
    Main function to process citations and display them with expanders.
    Uses direct Weaviate queries for reliable 1-to-1 chunk matching.
    Also stores citation data in session state for persistence.
    """
    # Parse citations from response
    citations = parse_citations_from_response(response_text)
    
    if not citations:
        return response_text, None
    
    # Extract chunk numbers
    chunk_numbers = extract_chunk_numbers(citations)
    
    if not chunk_numbers:
        return response_text, None
    
    # Get chunk data directly from Weaviate
    chunk_data = get_chunk_data_from_weaviate(chunk_numbers)
    
    if not chunk_data:
        st.warning("Could not find chunk data in Weaviate")
        return response_text, None
    
    # Create citation mapping
    citation_mapping = create_citation_mapping(citations, chunk_data)
    
    if not citation_mapping:
        return response_text, None
    
    # Store citation data in session state for persistence
    if 'all_citations' not in st.session_state:
        st.session_state.all_citations = {}
    
    # Add new citations to the persistent collection
    for chunk_num, citation_data in citation_mapping.items():
        citation_key = f"{citation_data['author']}_{citation_data['year']}_{chunk_num}"
        st.session_state.all_citations[citation_key] = {
            'chunk_num': chunk_num,
            'author': citation_data['author'],
            'year': citation_data['year'],
            'chunk_data': citation_data['chunk_data'],
            'timestamp': st.session_state.get('message_count', 0)
        }
    
    # Render citations with numbered format
    processed_text = render_citations_with_expanders(response_text, citation_mapping)
    
    return processed_text, citation_mapping

def display_persistent_citations():
    """
    Display all citations from the conversation in a persistent panel.
    """
    if 'all_citations' not in st.session_state or not st.session_state.all_citations:
        return
    
    st.markdown("---")
    st.markdown("### 📚 All Citations")
    
    # Sort citations by timestamp (most recent first)
    sorted_citations = sorted(
        st.session_state.all_citations.items(),
        key=lambda x: x[1]['timestamp'],
        reverse=True
    )
    
    # Group citations by author-year for better organization
    citation_groups = {}
    for citation_key, citation_info in sorted_citations:
        author_year = f"{citation_info['author']}, {citation_info['year']}"
        if author_year not in citation_groups:
            citation_groups[author_year] = []
        citation_groups[author_year].append(citation_info)
    
    # Display citations grouped by chunk number and document title
    for author_year, citations in citation_groups.items():
        for citation_info in citations:
            chunk_data = citation_info['chunk_data']
            with st.expander(f"📖 [{citation_info['chunk_num']}] {chunk_data['in_text_citation']}", expanded=False):
                st.markdown(f"**Document:** {chunk_data['document_title']}")
                st.markdown(f"**Page:** {chunk_data['page_number']}")
                st.markdown(f"**Bibliography:** {chunk_data['bibliography']}")
                st.markdown("---")
                st.markdown("**Content:**")
                st.markdown(chunk_data['content'])
                st.markdown("---")

def clear_all_citations():
    """
    Clear all stored citations from session state.
    """
    if 'all_citations' in st.session_state:
        st.session_state.all_citations = {}
        st.rerun()
