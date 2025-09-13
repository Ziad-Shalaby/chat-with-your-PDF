import os
import streamlit as st
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace
from langchain.chains import ConversationalRetrievalChain
from huggingface_hub import InferenceClient

# --------------------------- Config / Page ---------------------------
st.set_page_config(page_title="ChatWithYourDocs", page_icon="📄", layout="wide")
st.title("📄💬 Chat-With-Your-Docs (LangChain Edition)")
st.caption("Upload multiple PDF / DOCX / TXT files, then chat with them using RAG (Mistral via Hugging Face Inference API).")

# --------------------------- Load Hugging Face Token ---------------------------
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")

# --------------------------- Helpers: Document Loading ---------------------------
def load_document(uploaded_file):
    ext = Path(uploaded_file.name).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(uploaded_file)
    elif ext == ".txt":
        loader = TextLoader(uploaded_file)
    elif ext == ".docx":
        loader = Docx2txtLoader(uploaded_file)
    else:
        st.error(f"Unsupported file type: {ext}")
        return []
    return loader.load()

# --------------------------- Session State ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Chat model", ["mistralai/Mistral-7B-Instruct-v0.3"], index=0)
    chunk_size = st.slider("Chunk size (chars)", 500, 2000, 1000, 100)
    overlap = st.slider("Overlap (chars)", 50, 500, 200, 10)
    top_k = st.slider("Top-K retrieved chunks", 1, 10, 3, 1)
    max_new_tokens = st.slider("Max new tokens", 64, 1024, 300, 16)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

    if st.button("🧹 Reset chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared.")

# --------------------------- Upload & Index ---------------------------
uploaded_files = st.file_uploader("Upload one or more documents (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded in uploaded_files:
        docs = load_document(uploaded)
        documents.extend(docs)

    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        splits = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)

        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

        # HuggingFace chat client
        hf_client = InferenceClient(model=model_name, token=hf_token)
        llm = ChatHuggingFace(
            client=hf_client,
            model_kwargs={"max_new_tokens": max_new_tokens, "temperature": temperature}
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
        )

        st.session_state.qa_chain = qa_chain
        st.session_state.vector_ready = True
        st.success(f"✅ Indexed {len(splits)} chunks from {len(uploaded_files)} documents.")

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
            st.warning("Please upload and index documents first.")
        elif not hf_token:
            st.error("Hugging Face token not found. Please add it to your Streamlit secrets.")
        else:
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({
                    "question": user_msg,
                    "chat_history": [(m["content"], "") for m in st.session_state.chat_history if m["role"] == "user"]
                })

            answer = result["answer"]
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            with st.expander("🔎 Retrieved chunks (context)"):
                for i, doc in enumerate(result["source_documents"], 1):
                    st.markdown(f"**{i}. {doc.metadata.get('source', 'Unknown')}**\n\n{doc.page_content}")
