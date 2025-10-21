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

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="MediBot AI - Your Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize session state for chat management
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
    st.session_state.current_chat_id = None
    st.session_state.chat_counter = 0
if "initialization_done" not in st.session_state:
    st.session_state.initialization_done = False
if "lazy_load_triggered" not in st.session_state:
    st.session_state.lazy_load_triggered = False

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stChatMessage {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stChatMessage p, .stChatMessage div {
        color: #1a1a1a !important;
    }
    
    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        font-weight: 700;
        text-align: center;
        padding: 20px 0;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: 600;
    }
    
    .info-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: #1a1a1a;
    }
    
    .warning-box {
        background: rgba(255, 152, 0, 0.1);
        border-left: 4px solid #ff9800;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# LAZY INITIALIZATION - Only load when needed
def lazy_init_components():
    """Initialize heavy components only when user sends first message"""
    if st.session_state.initialization_done:
        return st.session_state.qa_chain
    
    if not st.session_state.lazy_load_triggered:
        return None
    
    progress_container = st.empty()
    
    with progress_container.container():
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Initialize Pinecone with timeout
            status_text.text("🔧 Connecting to database...")
            progress_bar.progress(20)
            
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Step 2: Load embeddings
            status_text.text("🧠 Loading AI model...")
            progress_bar.progress(40)
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
            )
            
            # Step 3: Initialize vectorstore
            status_text.text("📚 Connecting to knowledge base...")
            progress_bar.progress(60)
            
            index_name = "medical-chatbot-index"
            
            # Quick check if index exists
            existing_indexes = pc.list_indexes().names()
            
            if index_name not in existing_indexes:
                status_text.text("⚠️ Creating index (one-time setup)...")
                pc.create_index(
                    name=index_name,
                    dimension=384,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                # Wait for index to be ready
                max_wait = 60
                wait_time = 0
                while wait_time < max_wait:
                    if index_name in pc.list_indexes().names():
                        break
                    time.sleep(2)
                    wait_time += 2
            
            vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings,
                pinecone_api_key=PINECONE_API_KEY
            )
            
            # Step 4: Load LLM
            status_text.text("🚀 Initializing AI assistant...")
            progress_bar.progress(80)
            
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=512,
                timeout=30
            )
            
            # Create QA chain
            status_text.text("✅ Finalizing setup...")
            progress_bar.progress(95)
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True
            )
            
            st.session_state.qa_chain = qa_chain
            st.session_state.initialization_done = True
            
            progress_bar.progress(100)
            status_text.text("✅ Ready to chat!")
            time.sleep(0.5)
            
            return qa_chain
            
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)}")
            st.info("💡 **Troubleshooting tips:**\n"
                   "- Check your API keys in .env file\n"
                   "- Verify Pinecone index exists\n"
                   "- Check internet connection\n"
                   "- Try refreshing the page")
            return None
        finally:
            progress_container.empty()

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

# Function to update chat title
def update_chat_title(chat_id, first_message):
    if st.session_state.chat_sessions[chat_id]["title"].startswith("New Chat"):
        title = first_message[:30] + "..." if len(first_message) > 30 else first_message
        st.session_state.chat_sessions[chat_id]["title"] = title

# Sidebar
with st.sidebar:
    st.markdown("### 🏥 MediBot AI")
    st.markdown("---")
    
    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()
    
    st.markdown("---")
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
    st.markdown("#### 📊 Stats")
    
    if st.session_state.initialization_done:
        st.success("✅ AI Ready")
    else:
        st.warning("⏳ AI will load on first message")
    
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
    **MediBot AI** uses advanced AI to provide medical information.
    
    ⚠️ **Disclaimer**: Not a substitute for professional medical advice.
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

# Welcome message
if len(current_messages) == 0:
    st.markdown("""
    <div class="info-box">
        <h3 style="color: #667eea; margin-top: 0;">👋 Welcome to MediBot AI!</h3>
        <p style="font-size: 16px;">
            I'm here to help answer your medical questions using an extensive medical knowledge base. 
            Feel free to ask about symptoms, conditions, treatments, or general health information.
        </p>
        <p style="color: #666; font-size: 14px; margin-bottom: 0;">
            💡 <strong>Tip:</strong> The AI will initialize when you send your first message (takes ~10-15 seconds).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.initialization_done:
        st.markdown("""
        <div class="warning-box">
            <strong>⚡ Fast Load Mode:</strong> The AI assistant will load when you send your first message. 
            This makes the app open instantly!
        </div>
        """, unsafe_allow_html=True)
    
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

# Display chat history
for message in current_messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message and show_sources and message["sources"]:
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
    # Trigger lazy loading on first message
    if not st.session_state.lazy_load_triggered:
        st.session_state.lazy_load_triggered = True
    
    # Initialize if needed
    qa_chain = lazy_init_components()
    
    if qa_chain is None and not st.session_state.initialization_done:
        st.error("⚠️ Could not initialize the AI. Please check your API keys and try again.")
        st.stop()
    
    if len(current_messages) == 0:
        update_chat_title(st.session_state.current_chat_id, prompt)
    
    current_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Analyzing your question..."):
            try:
                response = st.session_state.qa_chain({"query": prompt})
                answer = response["result"]
                
                st.markdown(answer)
                
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
                
                current_messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources_list
                })
                
            except Exception as e:
                error_msg = f"😔 Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.info("💡 Try:\n- Rephrasing your question\n- Checking your internet connection\n- Refreshing the page if the error persists")
                current_messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 20px;">
    <p>⚠️ <strong>Medical Disclaimer:</strong> This chatbot provides general information only and is not a substitute for professional medical advice, diagnosis, or treatment.</p>
    <p style="font-size: 12px; opacity: 0.8;">Always consult with qualified healthcare professionals for medical concerns.</p>
</div>
""", unsafe_allow_html=True)