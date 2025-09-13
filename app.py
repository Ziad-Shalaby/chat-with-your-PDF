import os
import re
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import numpy as np
from pypdf import PdfReader
import docx
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --------------------------- Config / Page ---------------------------
st.set_page_config(page_title="ChatWithYourDocs", page_icon="📄", layout="wide")
st.title("📄💬 Chat-With-Your-Docs (LangChain RAG)")
st.caption("Upload PDF / DOCX / TXT files, then chat with them using RAG (LangChain + Hugging Face).")

# --------------------------- Load Hugging Face Token ---------------------------
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
if not hf_token:
    st.error("⚠️ HUGGINGFACEHUB_API_TOKEN not found in Streamlit secrets. Please add it to continue.")
    st.stop()

# --------------------------- Helpers: Read & Chunk ---------------------------
def read_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """Read and parse uploaded file content"""
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
    """Split text into clean sentences"""
    # Improved sentence splitting with better regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s and s.strip() and len(s.strip()) > 10]

def chunk_text(doc: Dict[str, Any], chunk_size: int = 120, overlap: int = 30) -> List[Dict[str, Any]]:
    """Create overlapping text chunks from document"""
    chunks, cid = [], 0
    
    for page in doc["pages"]:
        if not page["text"]:
            continue
            
        sentences = clean_and_split_sentences(page["text"])
        words = " ".join(sentences).split()
        
        if not words:
            continue
            
        step = max(1, chunk_size - overlap)
        start = 0
        
        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()
            
            if chunk_text and len(chunk_text) > 20:  # Filter very short chunks
                chunks.append({
                    "id": f"{doc['doc_id']}::chunk_{cid}",
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "page": page["page"],
                    "text": chunk_text
                })
                cid += 1
            
            start += step
            
            # Break if we've reached the end
            if end >= len(words):
                break
                
    return chunks

# --------------------------- LangChain: build vectorstore ---------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore_langchain(chunks: List[Dict[str, Any]], embed_model_name: str = "all-MiniLM-L6-v2"):
    """Build FAISS vectorstore from text chunks"""
    if not chunks:
        raise ValueError("No chunks provided to build the vectorstore.")

    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "source": c["source"], 
            "page": c.get("page", 1), 
            "doc_id": c["doc_id"], 
            "chunk_id": c["id"]
        }
        for c in chunks
    ]

    try:
        embeddings = SentenceTransformerEmbeddings(model_name=embed_model_name)
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        raise

# --------------------------- LangChain: build QA chain ---------------------------
@st.cache_resource(show_spinner=False)
def get_qa_chain(_vectorstore: FAISS, hf_token: str, model_name: str, top_k: int, system_instruction: str):
    """Create QA chain with custom prompt template"""
    retriever = _vectorstore.as_retriever(search_kwargs={"k": top_k})

    # Custom prompt template for better responses
    prompt_template = f"""
{system_instruction}

Context: {{context}}

Question: {{question}}

Answer: """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    try:
        # Determine the appropriate task based on the model
        if any(model in model_name.lower() for model in ["mistral", "mixtral", "zephyr"]):
            task = "conversational"
        else:
            task = "text-generation"

        # Pass parameters explicitly
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=hf_token,
            task=task,
            temperature=0.1,
            max_new_tokens=512,
            return_full_text=False
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        raise

# --------------------------- Session State ---------------------------
def initialize_session_state():
    """Initialize all session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "vector_ready" not in st.session_state:
        st.session_state.vector_ready = False
    if "current_files" not in st.session_state:
        st.session_state.current_files = []

initialize_session_state()

# --------------------------- Sidebar: Settings ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    model_name = st.selectbox(
        "Chat model (Hugging Face repo id)",
        [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "microsoft/DialoGPT-medium",
            "HuggingFaceH4/zephyr-7b-beta"
        ], 
        index=0
    )
    
    chunk_size = st.slider("Chunk size (words)", 50, 500, 120, 10)
    overlap = st.slider("Overlap (words)", 10, 100, 30, 5)
    top_k = st.slider("Top-K retrieved chunks", 1, 10, 3, 1)

    st.subheader("System Prompt")
    system_instruction = st.text_area(
        "Instruction for assistant", 
        value=(
            "You are a helpful assistant that answers questions based on the provided document context. "
            "Use only the information from the context to answer questions. "
            "If the answer cannot be found in the context, clearly state that you don't have enough information to answer."
        ),
        height=100
    )

    show_sources = st.checkbox("Show retrieved chunks", value=True)

    if st.button("🧹 Reset chat"):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 Clear documents"):
        st.session_state.vectorstore = None
        st.session_state.chunks = None
        st.session_state.vector_ready = False
        st.session_state.current_files = []
        st.session_state.chat_history = []
        st.rerun()

# --------------------------- Upload & Index ---------------------------
st.subheader("📁 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload one or more documents (PDF / DOCX / TXT)",
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    # Check if files have changed
    current_file_names = [f.name for f in uploaded_files]
    
    if current_file_names != st.session_state.current_files:
        st.session_state.current_files = current_file_names
        
        with st.spinner("Processing documents..."):
            all_chunks = []
            
            for uploaded in uploaded_files:
                doc = read_uploaded_file(uploaded)
                if doc["text"]:  # Only process if we got text
                    doc_chunks = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
                    all_chunks.extend(doc_chunks)

            if all_chunks:
                try:
                    vectorstore = build_vectorstore_langchain(all_chunks)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.chunks = all_chunks
                    st.session_state.vector_ready = True
                    
                    st.success(f"✅ Successfully indexed {len(all_chunks)} chunks from {len(uploaded_files)} documents.")
                    
                    # Show document stats
                    with st.expander("📊 Document Statistics"):
                        for file in uploaded_files:
                            file_chunks = [c for c in all_chunks if c["source"] == file.name]
                            st.write(f"**{file.name}**: {len(file_chunks)} chunks")
                            
                except Exception as e:
                    st.error(f"❌ Failed to build vectorstore: {e}")
                    st.session_state.vector_ready = False
            else:
                st.warning("⚠️ No valid text content found in uploaded files.")
                st.session_state.vector_ready = False

# --------------------------- Chat UI ---------------------------
st.subheader("💬 Chat with Your Documents")

# Display chat history
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Chat input
user_msg = st.chat_input("Ask about your documents...")

if user_msg:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    
    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        if not st.session_state.vector_ready or st.session_state.vectorstore is None:
            error_msg = "⚠️ Please upload and index documents first before asking questions."
            st.warning(error_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("Searching documents and generating response..."):
                try:
                    qa_chain = get_qa_chain(
                        st.session_state.vectorstore, 
                        hf_token, 
                        model_name, 
                        top_k, 
                        system_instruction
                    )
                    
                    result = qa_chain({"query": user_msg})
                    
                    # Extract answer and sources
                    if isinstance(result, dict) and "result" in result:
                        answer = result["result"].strip()
                        source_documents = result.get("source_documents", [])
                    else:
                        answer = str(result).strip()
                        source_documents = []
                    
                    # Clean up the answer
                    if not answer or answer.lower() in ["", "none", "n/a"]:
                        answer = "I couldn't find a relevant answer in the provided documents."
                    
                except Exception as e:
                    answer = f"❌ Error during response generation: {str(e)}"
                    source_documents = []
                    st.error(f"Error details: {e}")

            # Display answer
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            # Show sources if enabled and available
            if show_sources and source_documents:
                with st.expander(f"🔎 Retrieved Sources ({len(source_documents)} chunks)"):
                    for i, doc in enumerate(source_documents, 1):
                        meta = doc.metadata if hasattr(doc, "metadata") else {}
                        source = meta.get("source", "unknown")
                        page = meta.get("page", "?")
                        content = getattr(doc, "page_content", str(doc))
                        
                        st.markdown(f"**{i}. {source} (Page {page})**")
                        st.markdown(f"```\n{content}\n```")
                        st.markdown("---")

