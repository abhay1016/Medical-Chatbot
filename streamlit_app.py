import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="MediBot AI - Your Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern medical theme
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat container styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Chat message text color */
    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        color: #1a1a1a !important;
    }
    
    /* User message styling */
    [data-testid="stChatMessageContent"] {
        color: #1a1a1a !important;
    }
    
    /* Make all text in chat messages visible */
    .stMarkdown {
        color: #1a1a1a !important;
    }
    
    /* Title styling */
    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        font-weight: 700;
        text-align: center;
        padding: 20px 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Input box styling */
    .stChatInputContainer {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 25px;
        padding: 5px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Info box styling */
    .info-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Chat history item */
    .chat-history-item {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .chat-history-item:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
    }
    
    .active-chat {
        background: rgba(255, 255, 255, 0.3);
        border-left: 4px solid white;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize session state for chat management
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
    st.session_state.current_chat_id = None
    st.session_state.chat_counter = 0

# Initialize Pinecone
@st.cache_resource
def init_pinecone():
    return Pinecone(api_key=PINECONE_API_KEY)

pc = init_pinecone()

# Set up embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Load Pinecone index
index_name = "medical-chatbot-index"

# Check if index exists, create if not
if index_name not in pc.list_indexes().names():
    with st.spinner("🔧 Setting up medical knowledge base..."):
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        st.success("✅ Knowledge base initialized!")

# Connect to vector store
@st.cache_resource
def init_vectorstore():
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )

vectorstore = init_vectorstore()

# Load Groq model
@st.cache_resource
def load_model():
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=512
    )
    return llm

try:
    llm = load_model()
    
    # Set up RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    model_loaded = False

# Function to create new chat
def create_new_chat():
    chat_id = f"chat_{st.session_state.chat_counter}"
    st.session_state.chat_counter += 1
    st.session_state.chat_sessions[chat_id] = {
        "messages": [],
        "title": f"New Chat {st.session_state.chat_counter}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    st.session_state.current_chat_id = chat_id
    return chat_id

# Function to get current messages
def get_current_messages():
    if st.session_state.current_chat_id is None:
        create_new_chat()
    return st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"]

# Function to update chat title based on first message
def update_chat_title(chat_id, first_message):
    if st.session_state.chat_sessions[chat_id]["title"].startswith("New Chat"):
        # Use first 30 characters of first message as title
        title = first_message[:30] + "..." if len(first_message) > 30 else first_message
        st.session_state.chat_sessions[chat_id]["title"] = title

# Sidebar
with st.sidebar:
    st.markdown("### 🏥 MediBot AI")
    st.markdown("---")
    
    # New Chat Button (prominent)
    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()
    
    st.markdown("---")
    
    # Chat History
    st.markdown("#### 💬 Chat History")
    
    if len(st.session_state.chat_sessions) > 0:
        for chat_id, chat_data in reversed(list(st.session_state.chat_sessions.items())):
            is_active = chat_id == st.session_state.current_chat_id
            
            col1, col2 = st.columns([4, 1])
            
            with col1:
                if st.button(
                    f"{'🟢 ' if is_active else ''}{chat_data['title']}", 
                    key=f"chat_{chat_id}",
                    use_container_width=True
                ):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"del_{chat_id}"):
                    del st.session_state.chat_sessions[chat_id]
                    if st.session_state.current_chat_id == chat_id:
                        if len(st.session_state.chat_sessions) > 0:
                            st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[-1]
                        else:
                            st.session_state.current_chat_id = None
                    st.rerun()
    else:
        st.info("No chat history yet. Start a new chat!")
    
    st.markdown("---")
    
    # Session Stats
    st.markdown("#### 📊 Current Chat Stats")
    current_messages = get_current_messages()
    msg_count = len(current_messages)
    user_msgs = len([m for m in current_messages if m["role"] == "user"])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", msg_count)
    with col2:
        st.metric("Questions", user_msgs)
    
    st.markdown("---")
    
    st.markdown("#### ⚙️ Settings")
    show_sources = st.checkbox("Show Sources", value=True)
    
    st.markdown("---")
    
    st.markdown("#### ℹ️ About")
    st.info("""
    **MediBot AI** uses advanced AI to provide medical information based on a comprehensive knowledge base.
    
    ⚠️ **Disclaimer**: This is not a substitute for professional medical advice.
    """)
    
    if st.button("🗑️ Clear All Chats"):
        st.session_state.chat_sessions = {}
        st.session_state.current_chat_id = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("##### 🔒 Powered by:")
    st.markdown("• Groq AI (Llama 3.3)")
    st.markdown("• Pinecone Vector DB")
    st.markdown("• LangChain")

# Main content
st.markdown("<h1>🏥 MediBot AI - Your Medical Assistant</h1>", unsafe_allow_html=True)

# Create initial chat if none exists
if st.session_state.current_chat_id is None:
    create_new_chat()

current_messages = get_current_messages()

# Welcome message in a styled container
if len(current_messages) == 0:
    st.markdown("""
    <div class="info-box">
        <h3 style="color: #667eea; margin-top: 0;">👋 Welcome to MediBot AI!</h3>
        <p style="color: #666; font-size: 16px;">
            I'm here to help answer your medical questions using an extensive medical knowledge base. 
            Feel free to ask about symptoms, conditions, treatments, or general health information.
        </p>
        <p style="color: #999; font-size: 14px; margin-bottom: 0;">
            💡 <strong>Tip:</strong> Be specific with your questions for more accurate responses.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample questions
    st.markdown("### 🔍 Try asking:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🤒 What are symptoms of flu?"):
            st.session_state.sample_q = "What are the common symptoms of influenza?"
    
    with col2:
        if st.button("💊 Managing diabetes"):
            st.session_state.sample_q = "How can I manage type 2 diabetes?"
    
    with col3:
        if st.button("🏃 Exercise benefits"):
            st.session_state.sample_q = "What are the health benefits of regular exercise?"

if not model_loaded:
    st.error("⚠️ Model could not be loaded. Please check your configuration.")
    st.stop()

# Display chat history
for message in current_messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        
        # Display sources if available
        if message["role"] == "assistant" and "sources" in message and show_sources:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.markdown(source)
                    if i < len(message["sources"]):
                        st.markdown("---")

# Handle sample question
if "sample_q" in st.session_state:
    prompt = st.session_state.sample_q
    del st.session_state.sample_q
else:
    prompt = st.chat_input("💬 Ask me anything about health and medicine...")

# Process user input
if prompt:
    # Update chat title if this is the first message
    if len(current_messages) == 0:
        update_chat_title(st.session_state.current_chat_id, prompt)
    
    # Add user message
    current_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Analyzing your question..."):
            try:
                # Simulate thinking time for better UX
                time.sleep(0.5)
                
                response = qa_chain({"query": prompt})
                answer = response["result"]
                
                st.markdown(answer)
                
                # Prepare sources
                sources_list = []
                if response.get("source_documents") and show_sources:
                    with st.expander("📚 View Sources"):
                        for i, doc in enumerate(response["source_documents"], 1):
                            source_text = doc.page_content[:300] + "..."
                            sources_list.append(source_text)
                            st.markdown(f"**Source {i}:**")
                            st.markdown(source_text)
                            if i < len(response["source_documents"]):
                                st.markdown("---")
                
                # Add to chat history with sources
                current_messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources_list
                })
                
            except Exception as e:
                error_msg = f"😔 Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                current_messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 20px;">
    <p>⚠️ <strong>Medical Disclaimer:</strong> This chatbot provides general information only and is not a substitute for professional medical advice, diagnosis, or treatment.</p>
    <p style="font-size: 12px; opacity: 0.8;">Always consult with qualified healthcare professionals for medical concerns.</p>
</div>
""", unsafe_allow_html=True)