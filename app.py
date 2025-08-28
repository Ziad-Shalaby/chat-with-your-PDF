import os
import re
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

# -------------------------- Huggingface Key -------------------------
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
# --------------------------- Config / Page ---------------------------
st.set_page_config(page_title="ChatWithYourPDF", page_icon="📄", layout="wide")
st.title("📄💬 Chat-With-Your-PDF (Conversational)")
st.caption("Upload a PDF / DOCX / TXT, then ask questions grounded in the document. Uses a conversational model via Hugging Face Inference API.")

# --------------------------- Helpers: Embedding & Chunking ---------------------------
@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def read_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """Return a dict with doc_id, source, and extracted text."""
    ext = Path(uploaded_file.name).suffix.lower()
    text = ""

    if ext == ".pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

    elif ext == ".txt":
        # uploaded_file.read() returns bytes
        try:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
        except Exception:
            uploaded_file.seek(0)
            text = uploaded_file.read().decode("utf-8", errors="ignore")

    elif ext == ".docx":
        file_bytes = uploaded_file.read()
        document = docx.Document(BytesIO(file_bytes))
        text = "\n".join(p.text for p in document.paragraphs)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return {"doc_id": Path(uploaded_file.name).stem, "source": uploaded_file.name, "text": text.strip()}


def clean_and_split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s and s.strip()]


def chunk_text(doc: Dict[str, Any], chunk_size: int = 120, overlap: int = 30) -> List[Dict[str, Any]]:
    """Create overlapping word-based chunks (returns list of chunk dicts)."""
    sentences = clean_and_split_sentences(doc["text"]) if doc.get("text") else []
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

# --------------------------- Vector Store (FAISS) ---------------------------

def build_vector_store(chunks: List[Dict[str, Any]], embedder: SentenceTransformer):
    texts = [c["text"] for c in chunks]
    if not texts:
        raise ValueError("No text chunks to index.")
    # sentence-transformers encode -> numpy
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # normalize to use inner product as cosine-sim
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, texts


def search(query: str, index: faiss.IndexFlat, texts: List[str], embedder: SentenceTransformer, top_k: int = 5):
    q = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q)
    distances, indices = index.search(q, top_k)
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if 0 <= idx < len(texts):
            results.append({"text": texts[idx], "score": float(score)})
    return results

# --------------------------- Prompting / Chat messages ---------------------------

def format_history(history: List[Dict[str, str]], max_turns: int = 5) -> str:
    trimmed = history[-(max_turns * 2):]
    lines = []
    for m in trimmed:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def build_chat_messages(user_query: str, retrieved: List[Dict[str, Any]], history: List[Dict[str, str]], system_instruction: str = None) -> List[Dict[str, str]]:
    """Return a list of messages suitable for chat.completions endpoints.
    system_instruction: optional custom system prompt. If None, a strict default is used.
    """
    context = "\n".join([f"- {r['text']}" for r in retrieved]) if retrieved else "No relevant context found."
    hist = format_history(history, max_turns=5)

    default_system = (
        "You are a helpful assistant that MUST answer strictly using the provided document context. "
        "If the answer cannot be found in the context, say you don't know and optionally suggest a keyword to search for. "
        "Be concise and do not invent facts. Cite nothing outside the supplied context."
    )

    system_content = system_instruction if system_instruction else default_system
    user_prompt = (
        f"Chat History:\n{hist}\n\nDocument Context:\n{context}\n\nUser Question: {user_query}\n"
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]
    return messages

# --------------------------- Generation (conversational) ---------------------------

def generate_with_chat_completion(messages: List[Dict[str, str]], hf_token: str, model_name: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
    """Call the HF InferenceClient chat completion API and return assistant text.

    Notes: different providers may return slightly different shapes. We try
    to handle common variants (object with .choices, or dict with 'choices').
    """
    # set token env for libraries that rely on it
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    client = InferenceClient(token=hf_token)

    # prefer the chat completions path
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
    except Exception as e:
        # bubble up a helpful message
        raise RuntimeError(f"Chat completion request failed: {e}")

    # Parse result robustly
    assistant_text = ""
    # object-like with .choices (some SDKs)
    try:
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            # choices[0].message.content or choices[0].message['content']
            choice = resp.choices[0]
            if hasattr(choice, "message"):
                m = choice.message
                assistant_text = m.content if hasattr(m, "content") else m.get("content", "")
            else:
                # dict-like
                assistant_text = choice.get("message", {}).get("content", "")
    except Exception:
        assistant_text = ""

    if not assistant_text:
        # dict-like fallback
        try:
            data = resp if isinstance(resp, dict) else resp.__dict__
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or choices[0]
                assistant_text = (msg.get("content") or msg.get("text") or "").strip()
        except Exception:
            assistant_text = str(resp)

    return assistant_text.strip()

# --------------------------- RAG pipeline ---------------------------

def rag_answer(user_query: str, index: faiss.IndexFlat, texts: List[str], embedder: SentenceTransformer, history: List[Dict[str, str]], hf_token: str, model_name: str, top_k: int = 3, max_new_tokens: int = 300, temperature: float = 0.7):
    retrieved = search(user_query, index, texts, embedder, top_k=top_k)
    messages = build_chat_messages(user_query, retrieved, history)
    answer = generate_with_chat_completion(messages, hf_token, model_name, max_new_tokens, temperature)
    return answer, retrieved

# --------------------------- Session State ---------------------------
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

# --------------------------- Sidebar / Controls ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Chat model",
        options=[
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        index=0,
    )

    chunk_size = st.slider("Chunk size (words)", 80, 400, 120, 10)
    overlap = st.slider("Overlap (words)", 10, 200, 30, 5)
    top_k = st.slider("Top-K retrieved chunks", 1, 10, 3, 1)
    max_new_tokens = st.slider("Max new tokens", 64, 1024, 300, 16)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    show_sources = st.checkbox("Show retrieved chunks", value=True)
    st.markdown("---")
    if st.button("🧹 Reset chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared.")

# --------------------------- Upload & Index ---------------------------
col1, col2 = st.columns([1, 1])
with col1:
    uploaded = st.file_uploader("Upload a document (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])

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

with col2:
    st.subheader("📄 Document Status")
    if st.session_state.vector_ready:
        st.success(f"Ready · {st.session_state.doc_name}")
        st.write(f"Chunks in index: **{len(st.session_state.texts)}**")
    else:
        st.info("Upload a document to start.")

st.markdown("---")

# --------------------------- Chat UI ---------------------------
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
            st.error("Hugging Face token not found. Please add it to your Streamlit secrets.")
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
                        temperature=temperature,
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








