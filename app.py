import os
import re
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
import numpy as np
from pypdf import PdfReader
import docx
from langchain.schema import Document as LCDocument
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --------------------------- Config / Page ---------------------------
st.set_page_config(page_title="ChatWithYourDocs", page_icon="📄", layout="wide")
st.title("📄💬 Chat-With-Your-Docs (LangChain RAG)")
st.caption("Upload PDF / DOCX / TXT files, then chat with them using RAG (LangChain + Hugging Face).")

# --------------------------- Load Hugging Face Token ---------------------------
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")

# --------------------------- Helpers: Read & Chunk ---------------------------
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
    """Split document pages into overlapping chunks (word-based)"""
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

# --------------------------- LangChain: build vectorstore ---------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore_langchain(chunks: List[Dict[str, Any]], embed_model_name: str = "all-MiniLM-L6-v2"):
    """
    Convert chunks -> texts + metadatas, then build a FAISS vectorstore using
    SentenceTransformerEmbeddings (no HF token required for embeddings).
    """
    if not chunks:
        raise ValueError("No chunks provided to build the vectorstore.")

    texts = [c["text"] for c in chunks]
    metadatas = [
        {"source": c["source"], "page": c.get("page", None), "doc_id": c["doc_id"], "chunk_id": c["id"]}
        for c in chunks
    ]

    embeddings = SentenceTransformerEmbeddings(model_name=embed_model_name)
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore

# --------------------------- LangChain: build QA chain ---------------------------
@st.cache_resource(show_spinner=False)
def get_qa_chain(_vectorstore: FAISS, hf_token: str, model_name: str,
                 system_instruction: str, top_k: int,
                 max_new_tokens: int, temperature: float):
    """
    Build a RetrievalQA chain using HuggingFaceEndpoint LLM and the retriever from vectorstore.
    Cached to avoid rebuilding every query.
    """
    retriever = _vectorstore.as_retriever(search_kwargs={"k": top_k})

    # LLM wrapper using HuggingFaceEndpoint
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=hf_token,
        task="text-generation",   # ✅ Required
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )

    # Simple prompt
    prompt_template = (
        system_instruction.strip()
        + "\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer concisely using only the context. "
          "If answer not in the context, say you don't know."
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain

# --------------------------- Session State ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

# --------------------------- Sidebar: Settings ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Chat model (Hugging Face repo id)", ["mistralai/Mistral-7B-Instruct-v0.3"], index=0)
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

# --------------------------- Upload & Index (LangChain FAISS) ---------------------------
uploaded_files = st.file_uploader("Upload one or more documents (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    for uploaded in uploaded_files:
        doc = read_uploaded_file(uploaded)
        doc_chunks = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(doc_chunks)

    if all_chunks:
        try:
            vectorstore = build_vectorstore_langchain(all_chunks, embed_model_name="all-MiniLM-L6-v2")
            st.session_state.vectorstore = vectorstore
            st.session_state.chunks = all_chunks
            st.session_state.vector_ready = True
            st.success(f"✅ Indexed {len(all_chunks)} chunks from {len(uploaded_files)} documents.")
        except Exception as e:
            st.error(f"Failed to build vectorstore: {e}")

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
        if not st.session_state.vector_ready or st.session_state.vectorstore is None:
            st.warning("Please upload documents first.")
        elif not hf_token:
            st.error("Hugging Face token not found. Please add it to your Streamlit secrets.")
        else:
            with st.spinner("Thinking (LangChain + HuggingFace)..."):
                try:
                    qa_chain = get_qa_chain(
                        st.session_state.vectorstore,
                        hf_token,
                        model_name,
                        system_instruction,
                        top_k,
                        max_new_tokens,
                        temperature
                    )
                    res = qa_chain({"query": user_msg})
                    # RetrievalQA when return_source_documents=True returns a dict with 'result' and 'source_documents'
                    if isinstance(res, dict) and "result" in res:
                        answer = res["result"]
                        source_documents = res.get("source_documents", [])
                    else:
                        # backward compatible fallback
                        answer = str(res)
                        source_documents = []
                except Exception as e:
                    answer = f"❌ Error during generation: {e}"
                    source_documents = []

            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            if show_sources and source_documents:
                with st.expander("🔎 Retrieved chunks (context)"):
                    for i, doc in enumerate(source_documents, 1):
                        meta = doc.metadata if hasattr(doc, "metadata") else {}
                        source = meta.get("source", "unknown")
                        page = meta.get("page", "?")
                        # doc.page_content is the text chunk
                        content = getattr(doc, "page_content", str(doc))
                        st.markdown(f"**{i}. {source} · Page {page}**\n\n{content}")



