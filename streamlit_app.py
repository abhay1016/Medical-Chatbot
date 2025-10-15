import streamlit as st
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec

# -----------------------------------------------------------
# 🔧 Page Configuration
# -----------------------------------------------------------
st.set_page_config(
    page_title="MediBot AI - Your Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------
# 🎨 Custom Medical-Themed Styling (UI same as before)
# -----------------------------------------------------------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .stChatMessage {
        background: rgba(255,255,255,0.95);
        border-radius: 15px; padding: 15px; margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    .stChatMessage p, .stChatMessage div, .stChatMessage span,
    [data-testid="stChatMessageContent"], .stMarkdown {
        color: #1a1a1a !important;
    }
    h1 { color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-weight: 700; text-align: center; padding: 20px 0; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    .stChatInputContainer {
        background: rgba(255,255,255,0.1); border-radius: 25px; padding: 5px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 25px; padding: 10px 30px;
        font-weight: 600; transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    .streamlit-expanderHeader {
        background: rgba(102,126,234,0.1); border-radius: 10px; font-weight: 600;
    }
    .info-box {
        background: rgba(255,255,255,0.95); border-radius: 15px;
        padding: 20px; margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 🔑 Environment Variables
# -----------------------------------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------------------------------------
# 💾 Streamlit Session State Initialization
# -----------------------------------------------------------
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
    st.session_state.current_chat_id = None
    st.session_state.chat_counter = 0

# -----------------------------------------------------------
# ⚙️ CACHED INITIALIZATION BLOCKS
# -----------------------------------------------------------
with st.spinner("🚀 Initializing MediBot AI... Please wait a few seconds."):
    time.sleep(1)

@st.cache_resource(show_spinner="🔌 Connecting to Pinecone...")
def init_pinecone():
    return Pinecone(api_key=PINECONE_API_KEY)

pc = init_pinecone()

@st.cache_resource(show_spinner="🧠 Loading embeddings model...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

index_name = "medical-chatbot-index"
if index_name not in pc.list_indexes().names():
    with st.spinner("🧱 Setting up medical knowledge base..."):
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        st.success("✅ Knowledge base initialized!")

@st.cache_resource(show_spinner="🔗 Connecting to vector store...")
def init_vectorstore():
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )

vectorstore = init_vectorstore()

@st.cache_resource(show_spinner="⚙️ Loading Groq LLM model...")
def load_model():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=512
    )

llm = load_model()

@st.cache_resource(show_spinner="🔍 Building retrieval pipeline...")
def build_qa_chain(llm, vectorstore):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

qa_chain = build_qa_chain(llm, vectorstore)

# -----------------------------------------------------------
# 💬 Chat Management Helpers
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# 🧭 Sidebar
# -----------------------------------------------------------
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
                    st.session_state.current_chat_id = (
                        list(st.session_state.chat_sessions.keys())[-1]
                        if st.session_state.chat_sessions else None
                    )
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
    st.info("""
    **MediBot AI** uses advanced AI to provide medical insights.
    ⚠️ **Disclaimer**: This is not a substitute for professional medical advice.
    """)

    if st.button("🗑️ Clear All Chats"):
        st.session_state.chat_sessions.clear()
        st.session_state.current_chat_id = None
        st.rerun()

    st.markdown("---")
    st.markdown("##### 🔒 Powered by:")
    st.markdown("• Groq AI (Llama 3.3)")
    st.markdown("• Pinecone Vector DB")
    st.markdown("• LangChain")

# -----------------------------------------------------------
# 🧠 Main Chat Interface
# -----------------------------------------------------------
st.markdown("<h1>🏥 MediBot AI - Your Medical Assistant</h1>", unsafe_allow_html=True)
if st.session_state.current_chat_id is None:
    create_new_chat()

current_messages = get_current_messages()

# Welcome message
if len(current_messages) == 0:
    st.markdown("""
    <div class="info-box">
        <h3 style="color: #667eea; margin-top: 0;">👋 Welcome to MediBot AI!</h3>
        <p style="color: #666;">Ask about symptoms, treatments, or general health info.</p>
        <p style="color: #999; font-size: 14px;">💡 <strong>Tip:</strong> Be specific for better results.</p>
    </div>
    """, unsafe_allow_html=True)
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
        if message["role"] == "assistant" and "sources" in message and show_sources:
            with st.expander("📚 View Sources"):
                for i, src in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:** {src}")
                    if i < len(message["sources"]):
                        st.markdown("---")

# Input
if "sample_q" in st.session_state:
    prompt = st.session_state.sample_q
    del st.session_state.sample_q
else:
    prompt = st.chat_input("💬 Ask me anything about health and medicine...")

# Processing user input
if prompt:
    if len(current_messages) == 0:
        update_chat_title(st.session_state.current_chat_id, prompt)

    current_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Analyzing your question..."):
            try:
                time.sleep(0.5)
                response = qa_chain({"query": prompt})
                answer = response["result"]
                st.markdown(answer)

                sources_list = []
                if response.get("source_documents") and show_sources:
                    for doc in response["source_documents"]:
                        sources_list.append(doc.page_content[:300] + "...")
                current_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_list
                })
            except Exception as e:
                err = f"😔 Sorry, an error occurred: {str(e)}"
                st.error(err)
                current_messages.append({"role": "assistant", "content": err})

# -----------------------------------------------------------
# ⚠️ Footer Disclaimer
# -----------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:white; padding:20px;">
    <p><strong>Medical Disclaimer:</strong> MediBot AI provides general information only.</p>
    <p style="font-size:12px; opacity:0.8;">Always consult licensed professionals for medical advice.</p>
</div>
""", unsafe_allow_html=True)
