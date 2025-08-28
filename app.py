# app.py
# RAG Chatbot with Mistral-7B-Instruct via Hugging Face Inference API

import os
import re
from io import BytesIO
from pathlib import Path

import streamlit as st
import faiss
import numpy as np
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx


st.set_page_config(page_title="ChatWithYourPDF", page_icon="🧠", layout="wide")
st.title("🧠 Chat-With_Your-PDF")
st.caption("Upload a PDF/DOCX/TXT, then chat. Answers are grounded to retrieved chunks from your document.")

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# File reading
def read_uploaded_file(uploaded_file) -> dict:
    ext = Path(uploaded_file.name).suffix.lower()
    text = ""

    if ext == ".pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

    elif ext == ".txt":
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    elif ext == ".docx":
        file_bytes = uploaded_file.read()
        document = docx.Document(BytesIO(file_bytes))
        text = "\n".join(p.text for p in document.paragraphs)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return {"doc_id": Path(uploaded_file.name).stem, "source": uploaded_file.name, "text": text.strip()}

# Chunking
def clean_and_split_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s and s.strip()]

def chunk_text(doc: dict, chunk_size: int = 120, overlap: int = 30):
    sentences = clean_and_split_sentences(doc["text"])
    words = " ".join(sentences).split()
    chunks, start, cid = [], 0, 0
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append({
                "id": f"{doc['doc_id']}::chunk_{cid}",
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "text": chunk_text
            })
        cid += 1
        start += step
    return chunks

# Vector Store
def build_vector_store(chunks, embedder):
    texts = [c["text"] for c in chunks]
    if not texts:
        raise ValueError("No text chunks to index.")
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, texts

def search(query: str, index, texts, embedder, top_k: int = 5):
    q = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q)
    distances, indices = index.search(q, top_k)
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if 0 <= idx < len(texts):
            results.append({"text": texts[idx], "score": float(score)})
    return results

# Prompting (Mistral Instruct)
def format_history(history, max_turns=5):
    # keep last N turns (user+assistant pairs)
    trimmed = history[-(max_turns * 2):]
    lines = []
    for m in trimmed:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)

def build_mistral_prompt(user_query: str, retrieved, history):
    context = "\n".join([f"- {r['text']}" for r in retrieved]) if retrieved else "No relevant context found."
    hist = format_history(history, max_turns=5)

    system = (
        "You are a helpful assistant that answers STRICTLY using the provided document context. "
        "If the answer is not in the context, say you don't know and suggest what to search for next. "
        "Be concise and cite no external facts."
    )

    # Mistral Instruct style prompt
    prompt = f"""[INST] <<SYS>>
{system}
<</SYS>>

Chat History:
{hist}

Document Context:
{context}

User Question: {user_query}
[/INST]"""
    return prompt

# Generation via HF Inference API
def generate_with_mistral_inference(prompt: str, hf_token: str, model_name: str, max_new_tokens=300, temperature=0.7, top_p=0.9):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    client = InferenceClient(model=model_name, token=hf_token)
    # Stream=False: get final string
    text = client.text_generation(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.05,
        do_sample=True,
        stream=False,
    )
    return text.strip()

def rag_answer(user_query, index, texts, embedder, history, hf_token, model_name, top_k=3, max_new_tokens=300, temperature=0.7):
    retrieved = search(user_query, index, texts, embedder, top_k=top_k)
    prompt = build_mistral_prompt(user_query, retrieved, history)
    answer = generate_with_mistral_inference(
        prompt=prompt,
        hf_token=hf_token,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9
    )
    return answer, retrieved

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index" not in st.session_state:
    st.session_state.index = None
if "texts" not in st.session_state:
    st.session_state.texts = None
if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
    "LLaMA 3 model",
    options=[
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct"
    ],
    index=0
)
    hf_token = st.text_input("Hugging Face Token (hf_...)", type="password")
    chunk_size = st.slider("Chunk size (words)", 80, 300, 120, 10)
    overlap = st.slider("Overlap (words)", 10, 150, 30, 5)
    top_k = st.slider("Top-K retrieved chunks", 1, 10, 3, 1)
    max_new_tokens = st.slider("Max new tokens", 64, 1024, 300, 16)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    show_sources = st.checkbox("Show retrieved chunks", value=True)
    st.markdown("---")
    if st.button("🧹 Reset chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared.")

# Upload & Index
c1, c2 = st.columns([1, 1])

with c1:
    uploaded = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
    if uploaded:
        try:
            doc = read_uploaded_file(uploaded)
            if not doc["text"]:
                st.error("No text extracted from the document.")
            else:
                with st.spinner("Embedding & indexing..."):
                    embedder = get_embedder()
                    chunks = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
                    if not chunks:
                        st.error("0 chunks produced. Try adjusting chunk size/overlap.")
                    else:
                        index, texts = build_vector_store(chunks, embedder)
                        st.session_state.index = index
                        st.session_state.texts = texts
                        st.session_state.vector_ready = True
                        st.session_state.doc_name = uploaded.name
                        st.success(f"Indexed {len(chunks)} chunks from **{uploaded.name}**")
        except Exception as e:
            st.error(f"Failed to process file: {e}")

with c2:
    st.subheader("📄 Document Status")
    if st.session_state.vector_ready:
        st.success(f"Ready · {st.session_state.doc_name}")
        st.write(f"Chunks in index: **{len(st.session_state.texts)}**")
    else:
        st.info("Upload a document to start.")

st.markdown("---")

# Chat UI

# Show history
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_msg = st.chat_input("Ask about your document...")

if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        if not st.session_state.vector_ready:
            st.warning("Please upload a document first.")
        elif not hf_token:
            st.warning("Please enter your Hugging Face token in the sidebar.")
        else:
            with st.spinner("Thinking..."):
                try:
                    embedder = get_embedder()
                    answer, retrieved = rag_answer(
                        user_msg,
                        st.session_state.index,
                        st.session_state.texts,
                        embedder,
                        st.session_state.chat_history,
                        hf_token,
                        model_name,
                        top_k=top_k,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature
                    )
                except Exception as e:
                    answer = f"Generation failed: {e}"
                    retrieved = []

            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            if show_sources and retrieved:
                with st.expander("🔎 Retrieved chunks (context)"):
                    for i, r in enumerate(retrieved, 1):
                        st.markdown(f"**{i}. (score={r['score']:.3f})**\n\n{r['text']}")

