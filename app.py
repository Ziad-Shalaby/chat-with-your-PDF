import streamlit as st
from pathlib import Path
import re
from io import BytesIO
from pypdf import PdfReader
import docx
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


st.set_page_config(page_title="ChatWithYourFile", page_icon="📚", layout="wide")

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

def read_uploaded_file(uploaded_file) -> dict:
    """
    Read a Streamlit UploadedFile (pdf/txt/docx) and return a dict:
    { doc_id, source, text }
    """
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


def clean_and_split_sentences(text: str):
    # Simple sentence split; keeps punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s and s.strip()]


def chunk_text(doc: dict, chunk_size: int = 120, overlap: int = 30):
    """
    Word-level chunking with overlap.
    chunk_size/overlap are in words.
    """
    sentences = clean_and_split_sentences(doc["text"])
    words = " ".join(sentences).split()
    chunks = []
    start = 0
    chunk_id = 0

    if not words:
        return []

    step = max(1, chunk_size - overlap)

    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append({
                "id": f"{doc['doc_id']}::chunk_{chunk_id}",
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "text": chunk_text
            })
        chunk_id += 1
        start += step

    return chunks


def build_vector_store(chunks, embedder):
    texts = [c["text"] for c in chunks]
    if len(texts) == 0:
        raise ValueError("No text chunks to index.")
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # Normalize for cosine similarity via inner product
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


def format_chat_history_for_prompt(history, max_turns: int = 5):
    # Keep last max_turns (user+assistant pairs)
    trimmed = history[-(max_turns * 2):]
    lines = []
    for msg in trimmed:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def rag_answer(query, index, texts, embedder, generator, history, top_k=3, max_tokens=256):
    retrieved = search(query, index, texts, embedder, top_k=top_k)
    context = "\n\n".join([f"- {r['text']}" for r in retrieved]) if retrieved else "No relevant context found."
    prev = format_chat_history_for_prompt(history, max_turns=5)

    prompt = f"""You are a helpful assistant that answers strictly using the provided document context. 
If the answer is not in the context, say you don't know and suggest what to search for next.

Chat History:
{prev}

Document Context:
{context}

User Question: {query}

Answer:"""

    out = generator(prompt, max_length=max_tokens, do_sample=False)[0]["generated_text"].strip()
    return out, retrieved


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

if "index" not in st.session_state:
    st.session_state.index = None

if "texts" not in st.session_state:
    st.session_state.texts = None

if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

with st.sidebar:
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk size (words)", 80, 300, 120, 10)
    overlap = st.slider("Overlap (words)", 10, 150, 30, 5)
    top_k = st.slider("Top-K retrieved chunks", 1, 10, 3, 1)
    show_sources = st.checkbox("Show retrieved chunks", value=True)
    st.markdown("---")
    if st.button("🧹 Reset Chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")


st.title("📚 RAG Chatbot — Chat with Your Document")
st.caption("Upload a PDF/DOCX/TXT, then chat. The assistant answers using retrieved chunks from your file.")

col_left, col_right = st.columns([1, 1])

with col_left:
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        try:
            doc = read_uploaded_file(uploaded_file)
            if not doc["text"]:
                st.error("No text extracted from the document. Try a different file.")
            else:
                embedder = get_embedder()
                chunks = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
                if not chunks:
                    st.error("Document produced 0 chunks. Try reducing chunk size or uploading a different file.")
                else:
                    with st.spinner("Building vector index..."):
                        index, texts = build_vector_store(chunks, embedder)
                    st.session_state.index = index
                    st.session_state.texts = texts
                    st.session_state.vector_ready = True
                    st.session_state.doc_name = uploaded_file.name
                    st.success(f"Indexed {len(chunks)} chunks from **{uploaded_file.name}**")
        except Exception as e:
            st.error(f"Failed to process file: {e}")

with col_right:
    st.subheader("📄 Document Status")
    if st.session_state.vector_ready:
        st.success(f"Ready · {st.session_state.doc_name}")
        st.write(f"Chunks in index: **{len(st.session_state.texts)}**")
    else:
        st.info("Upload a document to start.")

st.markdown("---")


# Show previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_msg = st.chat_input("Ask about your document...")

if user_msg:
    # Echo user message
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        if not st.session_state.vector_ready:
            st.warning("Please upload a document first.")
        else:
            embedder = get_embedder()
            generator = get_generator()
            with st.spinner("Thinking..."):
                answer, retrieved = rag_answer(
                    user_msg,
                    st.session_state.index,
                    st.session_state.texts,
                    embedder,
                    generator,
                    st.session_state.chat_history,
                    top_k=top_k,
                    max_tokens=256
                )

            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            if show_sources and retrieved:
                with st.expander("🔎 Retrieved chunks (context)"):
                    for i, r in enumerate(retrieved, 1):
                        st.markdown(f"**{i}. (score={r['score']:.3f})**\n\n{r['text']}")
