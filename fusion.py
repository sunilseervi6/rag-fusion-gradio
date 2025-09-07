# -------------------------------
# Install dependencies (run once)
# -------------------------------
# !pip install faiss-cpu langchain-groq langchain-huggingface langchain gradio

import os
import gradio as gr
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from google.colab import userdata  # <-- Colab secrets

# -------------------------------
# Fetch Groq API key from Colab Secrets
# -------------------------------
try:
    GROQ_KEY = userdata.get("GROQ")  # Name of your secret in Colab
except Exception as e:
    raise ValueError("Failed to get Groq API key from Colab Secrets. Make sure the secret is set.") from e

# Set environment variable
os.environ["GROQ_API_KEY"] = GROQ_KEY

# Initialize LLM
llm = ChatGroq(api_key=GROQ_KEY, model="llama-3.3-70b-versatile", temperature=0)

# -------------------------------
# Reciprocal Rank Fusion function
# -------------------------------
def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [loads(doc) for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
    return reranked_results

# -------------------------------
# Build FAISS vector store from uploaded file
# -------------------------------
def build_faiss(file):
    text = open(file.name, "r", encoding="utf-8").read()
    documents = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_store = FAISS.from_documents(docs, embeddings)
    return docs, faiss_store

# -------------------------------
# Query expansion with Groq
# -------------------------------
query_prompt = PromptTemplate.from_template(
    "Generate 3 diverse search queries for the following question:\nQuestion: {original_query}\nQueries:"
)
generate_queries = query_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))

# -------------------------------
# Fusion RAG retrieval
# -------------------------------
def fusion_retrieve(query, retriever, docs, k=5):
    alt_queries = generate_queries.invoke({"original_query": query})
    all_results = []
    for q in [query] + alt_queries:
        retrieved_docs = retriever.similarity_search(q, k=k)
        all_results.append(retrieved_docs)
    fused_docs = reciprocal_rank_fusion(all_results)
    return fused_docs[:k]

# -------------------------------
# RAG answer function
# -------------------------------
def rag_fusion_answer(file, query):
    docs, faiss_store = build_faiss(file)
    fused_docs = fusion_retrieve(query, faiss_store, docs, k=5)
    context = "\n\n".join([doc.page_content for doc in fused_docs])
    prompt = f"Answer using only the context below.\nContext:\n{context}\nQuestion: {query}"
    response = llm.invoke(prompt)
    return response.content

# -------------------------------
# Gradio Interface
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Fusion RAG Chatbot (Colab Secrets)")
    file_input = gr.File(label="Upload a text file", file_types=[".txt"])
    query_input = gr.Textbox(label="Ask your question")
    output = gr.Textbox(label="Answer")

    def chat(file, query):
        if not file or not query.strip():
            return "Upload a file and type a question."
        return rag_fusion_answer(file, query)

    query_input.submit(chat, [file_input, query_input], output)
    gr.Button("Get Answer").click(chat, [file_input, query_input], output)

demo.launch()
