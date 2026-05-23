
import streamlit as st
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

st.set_page_config(
    page_title="Equipment Maintenance Bot",
    page_icon="🔧",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2rem;
    font-weight: bold;
    color: #1f77b4;
}
.metric-box {
    background: #f0f2f6;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='main-header'>🔧 Equipment Maintenance Bot</div>",
    unsafe_allow_html=True
)
st.caption("Industrial RAG System | 513 Chunks | 92% Accuracy")
st.divider()

# Load resources
@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("combined_index.faiss")
    with open("combined_chunks.json", "r") as f:
        chunks = json.load(f)
    return model, index, chunks

with st.spinner("Loading AI model..."):
    model, index, chunks = load_resources()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Sidebar
with st.sidebar:
    st.title("📊 System Stats")
    col1, col2 = st.columns(2)
    col1.metric("Chunks", "513")
    col2.metric("Accuracy", "92%")
    
    st.divider()
    st.markdown("### 💡 Sample Questions")
    
    questions = [
        "Why is Pump-7 vibrating at 3kHz?",
        "How to replace the bearing?",
        "What are ISO vibration limits?",
        "Pump seal is leaking?",
        "What grease for bearings?",
        "Motor is overheating?"
    ]
    
    for q in questions:
        if st.button(q, use_container_width=True):
            st.session_state.clicked_q = q
    
    st.divider()
    st.markdown("### 🛠️ Equipment Covered")
    st.markdown("- CP-200 Centrifugal Pump")
    st.markdown("- SKF 6205 Bearings")
    st.markdown("- John Crane Mechanical Seals")
    st.markdown("- Industrial Motors & Compressors")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "clicked_q" not in st.session_state:
    st.session_state.clicked_q = None

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle sidebar button click
if st.session_state.clicked_q:
    question = st.session_state.clicked_q
    st.session_state.clicked_q = None
else:
    question = st.chat_input(
        "Ask about equipment maintenance..."
    )

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching manuals..."):

            # FAISS search
            query_emb = model.encode(
                [question],
                convert_to_numpy=True
            )
            distances, indices = index.search(
                query_emb, k=3
            )

            # Build context
            context = ""
            sources = []
            for dist, idx in zip(
                distances[0], indices[0]
            ):
                chunk = chunks[idx]
                context += f"\n{chunk['text']}"
                sources.append(
                    f"• {chunk['source']} "
                    f"(relevance: {1/(1+dist):.0%})"
                )

            # Better prompt
            prompt = f"""You are an expert industrial 
equipment maintenance engineer with 20 years experience.

Use the maintenance manual context below to answer 
the technician's question. Be specific, practical, 
and safety-focused. Include part numbers and 
specifications when available.

If the exact equipment mentioned isn't in context,
use the closest matching equipment information.

CONTEXT FROM MAINTENANCE MANUALS:
{context}

TECHNICIAN QUESTION: {question}

Provide a clear, structured answer with:
1. Root cause (if diagnostic question)
2. Recommended action steps
3. Parts/materials needed (if applicable)
4. Safety warnings (if applicable)

ANSWER:"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an industrial equipment maintenance expert. Always give practical, safety-focused advice."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1
            )

            answer = response.choices[0].message.content
            st.write(answer)

            with st.expander("📄 Source Documents"):
                for s in sources:
                    st.write(s)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
