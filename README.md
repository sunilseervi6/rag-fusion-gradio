# rag-fusion-gradio
# 🧠 Fusion RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with **LangChain**, **Groq LLMs**, **Hugging Face embeddings**, and **FAISS**.  
The app allows you to upload a text file and ask questions, where the system retrieves relevant chunks from your file, fuses results using **Reciprocal Rank Fusion**, and generates answers with Groq’s **LLaMA 3.3 70B** model.  

---

## 🚀 Features
- 📂 **File Upload** – Upload any `.txt` file as knowledge source  
- 🔎 **Hybrid Retrieval** – Query expansion + Reciprocal Rank Fusion  
- ⚡ **Vector Search** – FAISS + Hugging Face `all-MiniLM-L6-v2` embeddings  
- 🤖 **LLM Powered** – Groq-hosted **LLaMA 3.3 70B** model for generation  
- 🌐 **Interactive UI** – Built with Gradio  

---

## 🛠️ Tech Stack
- **Python**
- **LangChain**
- **Groq API** (for LLM)
- **Hugging Face** (for embeddings)
- **FAISS** (vector store)
- **Gradio** (UI)

---

## 📦 Installation
Clone the repo and install dependencies:

```bash


# Install dependencies
pip install faiss-cpu langchain-groq langchain-huggingface langchain gradio

