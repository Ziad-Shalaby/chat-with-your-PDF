import os
import re
import pickle
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
import faiss
import numpy as np
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx

# --------------------------- Config / Page ---------------------------
st.set_page_config(page_title="ChatWithYourDocs", page_icon="📄", layout="wide")
st.title("📄💬 Chat-With-Your-Docs")
st.caption("Upload multiple PDF / DOCX / TXT files, then chat with them using RAG (Mistral via Hugging Face Inference API).")

# --------------------------- Load Hugging Face Token ---------------------------
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")

# --------------------------- Helpers: Embedding & Chunking ---------------------------
@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def read_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """Read PDF/DOCX/TXT into structured text pages"""
    ext = Path(uploaded_file.name).suffix.lower()
    text = ""
    pages = []

    try:
        if ext == ".pdf":
            reader = PdfReader(uploaded_file)
            for i, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append({"page": i, "text": page_text})
                    text += page_text + "\n"

        elif ext == ".txt":
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            pages = [{"page": 1, "text": content}]
            text = content

        elif ext == ".docx":
            file_bytes = uploaded_file.read()
            document = docx.Document(BytesIO(file_bytes))
            content = "\n".join(p.text for p in document.paragraphs if p.text.strip())
            pages = [{"page": 1, "text": content}]
            text = content

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    except Exception as e:
        st.error(f"❌ Error reading {uploaded_file.name}: {e}")
        return {"doc_id": Path(uploaded_file.name).stem, "source": uploaded_file.name, "text": "", "pages": []}

    return {
        "doc_id": Path(uploaded_file.name).stem,
        "source": uploaded_file.name,
        "text": text.strip(),
        "pages": pages
    }


def clean_and_split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s and s.strip()]


def chunk_text(doc: Dict[str, Any], chunk_size: int = 120, overlap: int = 30) -> List[Dict[str, Any]]:
    """Split document pages into overlapping chunks"""
    chunks, cid = [], 0
    for page in doc["pages"]:
        sentences = clean_and_split_sentences(page["text"]) if page["text"] else []
        words = " ".join(sentences).split()
        step = max(1, chunk_size - overlap)
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()
            if chunk_text:
                chunks.append({
                    "id": f"{doc['doc_id']}::chunk_{cid}",
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "page": page["page"],
                    "text": chunk_text
                })
            cid += 1
            start += step
    return chunks

# --------------------------- Vector Store (FAISS) ---------------------------
def build_vector_store(chunks: List[Dict[str, Any]], embedder: SentenceTransformer):
    texts = [c["text"] for c in chunks]
    if not texts:
        raise ValueError("No text chunks to index.")

    embeddings = embedder.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, texts, chunks


def search(query: str, index: faiss.IndexFlat, texts: List[str], chunks: List[Dict[str, Any]], embedder: SentenceTransformer, top_k: int = 5):
    q = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q)
    distances, indices = index.search(q, top_k)
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if 0 <= idx < len(texts):
            results.append({
                "text": texts[idx],
                "score": float(score),
                "source": chunks[idx]["source"],
                "page": chunks[idx].get("page", "?")
            })
    return results

# --------------------------- Prompting ---------------------------
def build_chat_messages(user_query: str, retrieved: List[Dict[str, Any]], history: List[Dict[str, str]], system_instruction: str) -> List[Dict[str, str]]:
    if retrieved:
        context = "\n".join([f"- {r['text']}" for r in retrieved])
    else:
        context = "No relevant context found."

    hist = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history[-10:]])

    user_prompt = (
        f"Chat History:\n{hist}\n\nDocument Context:\n{context}\n\nUser Question: {user_query}\n"
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt},
    ]
    return messages

# --------------------------- Generation ---------------------------
def generate_with_chat_completion(messages: List[Dict[str, str]], hf_token: str, model_name: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
    client = InferenceClient(token=hf_token)
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            return resp.choices[0].message.content.strip()
        return "⚠️ No response from model."
    except Exception as e:
        return f"❌ Generation failed: {e}"

# --------------------------- RAG pipeline ---------------------------
def rag_answer(user_query: str, index, texts, chunks, embedder, history, hf_token, model_name, system_instruction, top_k=3, max_new_tokens=300, temperature=0.7):
    retrieved = search(user_query, index, texts, chunks, embedder, top_k=top_k)
    if not retrieved:
        return "I couldn’t find anything related in your documents.", []
    messages = build_chat_messages(user_query, retrieved, history, system_instruction)
    answer = generate_with_chat_completion(messages, hf_token, model_name, max_new_tokens, temperature)
    return answer, retrieved

# --------------------------- Session State ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index" not in st.session_state:
    st.session_state.index = None
if "texts" not in st.session_state:
    st.session_state.texts = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Chat model", ["mistralai/Mistral-7B-Instruct-v0.3"], index=0)
    chunk_size = st.slider("Chunk size (words)", 80, 400, 120, 10)
    overlap = st.slider("Overlap (words)", 10, 200, 30, 5)
    top_k = st.slider("Top-K retrieved chunks", 1, 10, 3, 1)
    max_new_tokens = st.slider("Max new tokens", 64, 1024, 300, 16)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

    st.subheader("System Prompt")
    system_instruction = st.text_area("Instruction for assistant", value=(
        "You are a helpful assistant that MUST answer strictly using the provided document context. "
        "If the answer cannot be found in the context, say you don't know."
    ))

    show_sources = st.checkbox("Show retrieved chunks", value=True)

    if st.button("🧹 Reset chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared.")

    if st.session_state.chat_history:
        if st.download_button("💾 Download transcript", 
                              data="\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.chat_history]),
                              file_name="chat_transcript.txt"):
            st.success("Transcript downloaded.")

# --------------------------- Upload & Index ---------------------------
uploaded_files = st.file_uploader("Upload one or more documents (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    for uploaded in uploaded_files:
        doc = read_uploaded_file(uploaded)
        doc_chunks = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(doc_chunks)

    if all_chunks:
        embedder = get_embedder()
        index, texts, chunks = build_vector_store(all_chunks, embedder)
        st.session_state.index = index
        st.session_state.texts = texts
        st.session_state.chunks = chunks
        st.session_state.vector_ready = True
        st.success(f"✅ Indexed {len(all_chunks)} chunks from {len(uploaded_files)} documents.")

# --------------------------- Chat UI ---------------------------
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_msg = st.chat_input("Ask about your documents...")

if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        if not st.session_state.vector_ready:
            st.warning("Please upload documents first.")
        elif not hf_token:
            st.error("Hugging Face token not found. Please add it to your Streamlit secrets.")
        else:
            with st.spinner("Thinking..."):
                embedder = get_embedder()
                answer, retrieved = rag_answer(
                    user_msg,
                    st.session_state.index,
                    st.session_state.texts,
                    st.session_state.chunks,
                    embedder,
                    st.session_state.chat_history,
                    hf_token,
                    model_name,
                    system_instruction,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            if show_sources and retrieved:
                with st.expander("🔎 Retrieved chunks (context)"):
                    for i, r in enumerate(retrieved, 1):
                        st.markdown(f"**{i}. {r['source']} · Page {r['page']} (score={r['score']:.3f})**\n\n{r['text']}")
