# 🔧 Equipment Maintenance RAG Chatbot

Industrial NLP chatbot for equipment maintenance using RAG.

## 🎯 Project Overview
- Domain: Manufacturing / Oil & Gas
- Accuracy: 92% on test set (10/10 questions passed)
- Dataset: 513 chunks (pump manuals + maintenance logs)
- Model: all-MiniLM-L6-v2 + Groq LLaMA 3.1

## 🏗️ Architecture
User Query → Embedding Model → FAISS Search → LLM → Answer

## 📊 Tech Stack
- Embeddings: sentence-transformers
- Vector DB: FAISS
- LLM: Groq LLaMA 3.1 8B
- UI: Streamlit
- Deployment: HuggingFace Spaces

## 📈 Evaluation Results
- Accuracy: 92%
- Pass Rate: 10/10
- Chunks Indexed: 513
- Avg Response Time: less than 3 sec

## 🚀 Live Demo
https://huggingface.co/spaces/Tanveer5398/equipment-rag-bot

## 👨‍💻 Built By
Tanveer — NLP Engineer
10-day project: Industrial RAG Chatbot