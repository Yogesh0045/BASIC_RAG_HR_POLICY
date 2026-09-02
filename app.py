"""Streamlit chat app for the HR Policy Assistant.
Run with: streamlit run app.py
"""

import streamlit as st
from hr_assistant.pipeline import setup_rag_pipeline
from hr_assistant.agent import ask_assistant

from hr_assistant.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .header-title {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subheader-text {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
    st.session_state.pipeline_initialized = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "error_message" not in st.session_state:
    st.session_state.error_message = None


def initialize_pipeline():
    """Initialize the RAG pipeline."""
    logger.info("Initializing HR RAG pipeline")
    try:
        with st.spinner("🔄 Setting up HR RAG Pipeline..."):
            st.session_state.agent_executor, _, _ = setup_rag_pipeline()
            st.session_state.pipeline_initialized = True
            st.session_state.error_message = None
            logger.info("HR RAG pipeline initialized successfully")
    except Exception as e:
        logger.exception("Failed to initialize HR RAG pipeline")
        st.session_state.error_message = f"Error initializing pipeline: {str(e)}"
        st.error(st.session_state.error_message)


def process_question(question: str) -> str:
    """Process a question through the agent."""
    logger.info("Processing user question")
    try:
        if not st.session_state.pipeline_initialized:
            logger.warning("Question received before pipeline initialization")
            return "Error: Pipeline not initialized. Please refresh the page."
        
        response = ask_assistant(st.session_state.agent_executor, question)
        logger.info("User question processed successfully")
        return response
    except Exception as e:
        logger.exception("Failed to process user question")
        error_msg = f"Error processing question: {str(e)}"
        return error_msg


# Main UI
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="header-title">💼 HR Policy Assistant</div>', unsafe_allow_html=True)
with col2:
    if st.button("🔄 Refresh Pipeline", key="refresh_btn"):
        logger.info("Pipeline refresh requested")
        st.session_state.pipeline_initialized = False
        st.session_state.chat_history = []
        initialize_pipeline()
        st.rerun()

st.markdown('<div class="subheader-text">Ask questions about HR policies, leave, benefits, and more</div>', unsafe_allow_html=True)

# Initialize pipeline on first load
if not st.session_state.pipeline_initialized:
    initialize_pipeline()

# Display error message if any
if st.session_state.error_message:
    st.error(st.session_state.error_message)
else:
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    st.markdown("---")
    user_input = st.chat_input("Ask me anything about HR policies...", key="chat_input")
    
    if user_input:
        logger.info("Chat message received")
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process and get response
        with st.chat_message("assistant"):
            with st.spinner("Searching HR policies..."):
                response = process_question(user_input)
        
        # Add assistant message to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Rerun to update chat display
        st.rerun()

# Sidebar information
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    ### About this Assistant
    This HR Policy Assistant uses RAG (Retrieval-Augmented Generation) 
    to answer questions about company HR policies.
    
    ### How it works
    1. Your question is processed by an AI agent
    2. The agent searches the HR policy documents
    3. Results are used to generate an accurate answer
    
    ### Topics Covered
    - Leave policies
    - Work from home policies
    - Probation periods
    - Notice periods
    - Reimbursement policies
    - Code of conduct
    - Holiday schedules
    - Exit procedures
    
    ### Tips
    - Ask specific questions for better results
    - If the answer isn't found, it will say so
    - You can ask follow-up questions
    """)
    
    st.divider()
    
    # Chat management
    st.subheader("💬 Chat Management")
    if st.button("🗑️ Clear Chat History", key="clear_history_btn"):
        logger.info("Chat history cleared")
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
        st.rerun()
    
    # Statistics
    st.divider()
    st.subheader("📊 Statistics")
    st.metric("Total Messages", len(st.session_state.chat_history))
    st.metric("Pipeline Status", "✅ Ready" if st.session_state.pipeline_initialized else "⏳ Loading...")
