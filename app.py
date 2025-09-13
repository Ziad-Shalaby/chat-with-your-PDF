import os
from pathlib import Path
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceChat
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# --------------------------- Config / Page ---------------------------
st.set_page_config(page_title="ChatWithYourDocs", page_icon="📄", layout="wide")
st.title("📄💬 Chat-With-Your-Docs (LangChain Edition)")
st.caption("Upload multiple PDF / DOCX / TXT files, then chat with them using LangChain + Hugging Face Inference API.")

# --------------------------- Load Hugging Face Token ---------------------------
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")

# --------------------------- Embeddings ---------------------------
@st.cache_resource
def get_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --------------------------- File Loader ---------------------------
def load_file(uploaded_file):
    ext = Path(uploaded_file.name).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(uploaded_file)
    elif ext == ".docx":
        loader = Docx2txtLoader(uploaded_file)
    elif ext == ".txt":
        loader = TextLoader(uploaded_file, encoding="utf-8")
    else:
        st.error(f"Unsupported file type: {ext}")
        return []
    return loader.load()

# --------------------------- Process Documents ---------------------------
def process_docs(uploaded_files, chunk_size=500, overlap=100):
    docs = []
    for f in uploaded_files:
        docs.extend(load_file(f))
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

# --------------------------- Vector Store ---------------------------
def build_vector_store(docs, embedder):
    return FAISS.from_documents(docs, embedder)

# --------------------------- RAG Chain ---------------------------
def get_rag_chain(vectorstore, model_name, hf_token):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = HuggingFaceChat(
        model=model_name,
        huggingfacehub_api_token=hf_token,
        max_new_tokens=300,
        temperature=0.7,
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return chain

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Chat model", ["mistralai/Mistral-7B-Instruct-v0.3"], index=0)
    chunk_size = st.slider("Chunk size (characters)", 200, 2000, 500, 50)
    overlap = st.slider("Overlap (characters)", 50, 500, 100, 10)
    top_k = st.slider("Top-K retrieved chunks", 1, 10, 3, 1)
    if st.button("🧹 Reset chat"):
        st.session_state.pop("qa_chain", None)
        st.session_state.pop("vectorstore", None)
        st.session_state.pop("chat_history", None)
        st.success("Chat cleared.")

# --------------------------- Upload & Build ---------------------------
uploaded_files = st.file_uploader("Upload your documents (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files and "vectorstore" not in st.session_state:
    with st.spinner("Processing documents..."):
        docs = process_docs(uploaded_files, chunk_size=chunk_size, overlap=overlap)
        embedder = get_embedder()
        vectorstore = build_vector_store(docs, embedder)
        st.session_state.vectorstore = vectorstore
        st.session_state.qa_chain = get_rag_chain(vectorstore, model_name, hf_token)
        st.success(f"✅ Indexed {len(docs)} chunks from {len(uploaded_files)} documents.")

# --------------------------- Chat UI ---------------------------
if "qa_chain" in st.session_state:
    qa_chain = st.session_state.qa_chain

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_msg = st.chat_input("Ask about your documents...")
    if user_msg:
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = qa_chain({"question": user_msg})
                answer = result["answer"]
                sources = result.get("source_documents", [])

            st.write(answer)
            if sources:
                with st.expander("🔎 Retrieved chunks (context)"):
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**{i}. {doc.metadata.get('source','?')}**\n\n{doc.page_content}")

        # Save chat
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("👆 Upload some documents to start chatting.")
