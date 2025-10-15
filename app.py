import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from pinecone import Pinecone
import time
from datetime import datetime

st.set_page_config(
    page_title="MediBot AI - Your Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        backdrop-filter: blur(10px);
    }
    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        color: #1a1a1a !important;
    }
    [data-testid="stChatMessageContent"] {
        color: #1a1a1a !important;
    }
    .stMarkdown {
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
    .stChatInputContainer {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 25px;
        padding: 5px;
    }
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
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        font-weight: 600;
    }
    .info-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY or not GROQ_API_KEY:
    st.error("⚠️ Missing API keys. Check your environment variables.")
    st.stop()

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
    st.session_state.current_chat_id = None
    st.session_state.chat_counter = 0

@st.cache_resource(show_spinner=False)
def init_resources():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = PineconeVectorStore(
            index_name="medical-chatbot-index",
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=512,
            timeout=30,
            max_retries=2
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        return None

with st.spinner("🔄 Loading MediBot AI..."):
    qa_chain = init_resources()

if qa_chain is None:
    st.error("⚠️ Failed to initialize. Please refresh or check configuration.")
    st.stop()

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

def get_current_messages():
    if st.session_state.current_chat_id is None:
        create_new_chat()
    return st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"]

def update_chat_title(chat_id, first_message):
    if st.session_state.chat_sessions[chat_id]["title"].startswith("New Chat"):
        title = first_message[:30] + "..." if len(first_message) > 30 else first_message
        st.session_state.chat_sessions[chat_id]["title"] = title

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
    st.info("**MediBot AI** uses advanced AI to provide medical information.\n\n⚠️ **Disclaimer**: Not a substitute for professional medical advice.")
    
    if st.button("🗑️ Clear All Chats"):
        st.session_state.chat_sessions = {}
        st.session_state.current_chat_id = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("##### 🔒 Powered by:")
    st.markdown("• Groq AI (Llama 3.3)")
    st.markdown("• Pinecone Vector DB")
    st.markdown("• LangChain")

st.markdown("<h1>🏥 MediBot AI - Your Medical Assistant</h1>", unsafe_allow_html=True)

if st.session_state.current_chat_id is None:
    create_new_chat()

current_messages = get_current_messages()

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

for message in current_messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message and show_sources:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.markdown(source)
                    if i < len(message["sources"]):
                        st.markdown("---")

if "sample_q" in st.session_state:
    prompt = st.session_state.sample_q
    del st.session_state.sample_q
else:
    prompt = st.chat_input("💬 Ask me anything about health and medicine...")

if prompt:
    if len(current_messages) == 0:
        update_chat_title(st.session_state.current_chat_id, prompt)
    
    current_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Analyzing your question..."):
            try:
                response = qa_chain({"query": prompt})
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
                current_messages.append({"role": "assistant", "content": error_msg})

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 20px;">
    <p>⚠️ <strong>Medical Disclaimer:</strong> This chatbot provides general information only and is not a substitute for professional medical advice, diagnosis, or treatment.</p>
    <p style="font-size: 12px; opacity: 0.8;">Always consult with qualified healthcare professionals for medical concerns.</p>
</div>
""", unsafe_allow_html=True)