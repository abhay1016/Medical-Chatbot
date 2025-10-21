import streamlit as st
import os
import requests
from dotenv import load_dotenv
from datetime import datetime

st.set_page_config(page_title="MediBot AI", page_icon="🏥", layout="wide")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ Set GROQ_API_KEY in environment variables")
    st.stop()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Minimal styling
st.markdown("""
<style>
    .stApp {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
    h1 {color: white; text-align: center;}
    [data-testid="stSidebar"] {background: #667eea;}
</style>
""", unsafe_allow_html=True)

def query_groq(question, context=""):
    """Direct Groq API call - NO embeddings needed!"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Medical-focused system prompt
    system_prompt = """You are MediBot, an AI medical assistant. Provide accurate, evidence-based medical information.
    
Important guidelines:
- Give clear, accurate medical information
- Explain medical terms in simple language
- Always remind users to consult healthcare professionals
- If unsure, say so clearly
- Focus on general health education
- Be empathetic and supportive"""

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Sidebar
with st.sidebar:
    st.markdown("### 🏥 MediBot AI")
    st.markdown("**Ultra-Light Version**")
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("#### 📊 Stats")
    st.metric("Messages", len(st.session_state.messages))
    
    st.markdown("---")
    st.info("""
    **⚡ Ultra-Fast Mode**
    
    No vector database needed!
    Direct AI responses.
    
    ⚠️ Not medical advice.
    """)

# Main content
st.markdown("<h1>🏥 MediBot AI - Ultra Light</h1>", unsafe_allow_html=True)

if len(st.session_state.messages) == 0:
    st.markdown("""
    <div style="background: white; border-radius: 15px; padding: 20px; margin: 20px 0;">
        <h3 style="color: #667eea;">👋 Welcome!</h3>
        <p>Ask me anything about health and medicine.</p>
        <p style="color: #666;"><strong>⚡ Lightning fast:</strong> No loading time!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🤒 Cold & Flu"):
            st.session_state.quick = "What are the differences between a cold and flu?"
    with col2:
        if st.button("💊 Headache Relief"):
            st.session_state.quick = "What are effective ways to relieve headaches?"
    with col3:
        if st.button("🏃 Stay Healthy"):
            st.session_state.quick = "What are the best ways to stay healthy?"

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle input
if "quick" in st.session_state:
    prompt = st.session_state.quick
    del st.session_state.quick
else:
    prompt = st.chat_input("💬 Ask anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Thinking..."):
            response = query_groq(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white;">
    <small>⚠️ For information only. Not medical advice. Consult healthcare professionals.</small>
</div>
""", unsafe_allow_html=True)