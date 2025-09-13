import os
from io import BytesIO
from pathlib import Path
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEndpoint

# --------------------------- Config / Page ---------------------------
st.set_page_config(page_title="ChatWithYourDocs (LangChain)", page_icon="📄", layout="wide")
st.title("📄💬 Chat-With-Your-Docs (LangChain)")
st.caption("Upload multiple PDF / DOCX / TXT files, then chat with them using RAG (LangChain + Mistral)")

# --------------------------- Hugging Face Token ---------------------------
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")

# --------------------------- Sidebar Settings ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Chat model", ["mistralai/Mistral-7B-Instruct-v0.3"], index=0)
    chunk_size = st.slider("Chunk size (chars)", 300, 2000, 1000, 100)
    overlap = st.slider("Overlap (chars)", 50, 500, 200, 50)
    top_k = st.slider("Top-K retrieved chunks", 1, 10, 3, 1)

    st.subheader("System Prompt")
    system_instruction = st.text_area("Instruction for assistant", value=(
        "You are a helpful assistant that MUST answer strictly using the provided document context. "
        "If the answer cannot be found in the context, say you don't know."
    ))

    if st.button("🧹 Reset chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared.")

# --------------------------- Session State ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --------------------------- File Upload ---------------------------
uploaded_files = st.file_uploader(
    "Upload one or more documents (PDF / DOCX / TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    docs = []
    for uploaded in uploaded_files:
        ext = Path(uploaded.name).suffix.lower()
        temp_path = f"temp_{uploaded.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())

        if ext == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif ext == ".txt":
            loader = TextLoader(temp_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(temp_path)
        else:
            st.error(f"Unsupported file type: {ext}")
            continue

        docs.extend(loader.load())

    if docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        split_docs = splitter.split_documents(docs)

        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(split_docs, embedder)

        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            token=hf_token,
            task="text-generation",
            max_new_tokens=300,
            temperature=0.7,
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
        )

        st.session_state.qa_chain = qa_chain
        st.success(f"✅ Indexed {len(split_docs)} chunks from {len(uploaded_files)} documents.")

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
        if not st.session_state.qa_chain:
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

                if "source_documents" in result:
                    with st.expander("🔎 Retrieved chunks (context)"):
                        for i, doc in enumerate(result["source_documents"], 1):
                            st.markdown(f"**{i}. {doc.metadata.get('source', 'unknown')}**\n\n{doc.page_content}")
