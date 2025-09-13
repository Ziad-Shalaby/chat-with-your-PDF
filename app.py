import os
import streamlit as st
from pathlib import Path
from io import BytesIO

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA

# --------------------------- Config ---------------------------
st.set_page_config(page_title="ChatWithYourDocs", page_icon="📄", layout="wide")
st.title("📄💬 Chat-With-Your-Docs (LangChain)")
st.caption("Upload PDFs, DOCX, or TXT files. Ask questions using RAG with Mistral via Hugging Face Inference API.")

# Hugging Face Token
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")

# --------------------------- Helpers ---------------------------
def load_file(uploaded_file):
    ext = Path(uploaded_file.name).suffix.lower()
    tmp_path = f"/tmp/{uploaded_file.name}"

    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())

    if ext == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(tmp_path)
    elif ext == ".txt":
        loader = TextLoader(tmp_path, encoding="utf-8")
    else:
        st.error(f"❌ Unsupported file type: {ext}")
        return []

    return loader.load()


def create_vector_store(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def create_rag_chain(vectorstore, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=hf_token,
        temperature=0.7,
        max_new_tokens=300
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk size", 200, 1000, 500, 50)
    overlap = st.slider("Overlap", 50, 300, 100, 10)
    top_k = st.slider("Top-K Chunks", 1, 10, 3, 1)
    max_new_tokens = st.slider("Max new tokens", 64, 1024, 300, 16)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

    if st.button("🧹 Reset chat"):
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.success("Chat cleared.")

# --------------------------- Upload & Index ---------------------------
uploaded_files = st.file_uploader("Upload documents (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    docs = []
    for file in uploaded_files:
        docs.extend(load_file(file))

    if docs:
        st.success(f"✅ Loaded {len(docs)} document chunks.")
        st.session_state.vectorstore = create_vector_store(docs, chunk_size, overlap)

# --------------------------- Chat ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.session_state.get("vectorstore"):
    qa_chain = create_rag_chain(st.session_state.vectorstore)

    user_input = st.chat_input("Ask something about your documents...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({"query": user_input})
                answer = response["result"]
                sources = response.get("source_documents", [])

            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            if sources:
                with st.expander("🔎 Sources"):
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**{i}. {doc.metadata.get('source','')}**\n\n{doc.page_content[:500]}...")

else:
    st.info("⬆️ Upload documents to start chatting.")
