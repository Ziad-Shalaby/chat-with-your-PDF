# app.py

import streamlit as st
from pathlib import Path
import re
from pypdf import PdfReader
import docx
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def load_file(file):
    ext = Path(file.name).suffix.lower()
    text = ""

    if ext == ".pdf":
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"

    elif ext == ".txt":
        text = file.read().decode("utf-8")

    elif ext == ".docx":
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])

    else:
        st.error(f"Unsupported file type: {ext}")
        return None

    return {"doc_id": Path(file.name).stem, "source": file.name, "text": text}

def clean_and_split_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(doc, chunk_size=120, overlap=30):
    sentences = clean_and_split_sentences(doc["text"])
    words = " ".join(sentences).split()
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "id": f"{doc['doc_id']}::chunk_{chunk_id}",
            "doc_id": doc["doc_id"],
            "source": doc["source"],
            "text": chunk_text
        })
        chunk_id += 1
        start += chunk_size - overlap
    return chunks

def build_vector_store(chunks, embedder):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, texts

def search(query, index, texts, embedder, top_k=5):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, top_k)
    return [{"text": texts[i], "score": float(score)} for i, score in zip(indices[0], distances[0])]

def rag_answer(query, index, texts, embedder, generator, top_k=3, max_tokens=200):
    retrieved = search(query, index, texts, embedder, top_k=top_k)
    context = "\n".join([r["text"] for r in retrieved])
    prompt = f"""You are an assistant that answers questions using the given context.

Context:
{context}

Question: {query}
Answer:"""
    result = generator(prompt, max_length=max_tokens, do_sample=False)[0]["generated_text"]
    return result

# Streamlit App:

st.title("📚 RAG Document Q&A App")
st.write("Upload a document (PDF, TXT, DOCX), and ask questions about it using Retrieval-Augmented Generation.")

# File Upload
uploaded_file = st.file_uploader("Upload your file", type=["pdf", "txt", "docx"])

if uploaded_file:
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    
    # Process file
    with st.spinner("Processing document..."):
        doc = load_file(uploaded_file)
        chunks = chunk_text(doc)
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        index, texts = build_vector_store(chunks, embedder)
        generator = pipeline("text2text-generation", model="google/flan-t5-base")
    st.success(f"Document processed into {len(chunks)} chunks.")

    # Question Input
    query = st.text_input("Ask a question about the document:")
    if query:
        with st.spinner("Generating answer..."):
            answer = rag_answer(query, index, texts, embedder, generator)
        st.subheader("Answer:")
        st.write(answer)
