import os
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any
from io import BytesIO

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.schema import BaseMessage, HumanMessage, AIMessage

# Additional imports for file processing
from pypdf import PdfReader
import docx
import tempfile

# --------------------------- Config / Page ---------------------------
st.set_page_config(page_title="ChatWithYourDocs - LangChain", page_icon="📄", layout="wide")
st.title("📄💬 Chat-With-Your-Docs (LangChain)")
st.caption("Upload multiple PDF / DOCX / TXT files, then chat with them using LangChain RAG pipeline.")

# --------------------------- Load Hugging Face Token ---------------------------
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")

# --------------------------- Document Processing ---------------------------
def process_uploaded_files(uploaded_files, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Process uploaded files and return LangChain Documents"""
    documents = []
    
    for uploaded_file in uploaded_files:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        try:
            if file_extension == ".pdf":
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Use PyPDFLoader
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                # Add source metadata
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                    doc.metadata["file_type"] = "pdf"
                
                documents.extend(docs)
                os.unlink(tmp_path)  # Clean up temp file
                
            elif file_extension == ".txt":
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as tmp_file:
                    content = uploaded_file.read().decode("utf-8", errors="ignore")
                    tmp_file.write(content)
                    tmp_path = tmp_file.name
                
                # Use TextLoader
                loader = TextLoader(tmp_path, encoding='utf-8')
                docs = loader.load()
                
                # Add source metadata
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                    doc.metadata["file_type"] = "txt"
                
                documents.extend(docs)
                os.unlink(tmp_path)  # Clean up temp file
                
            elif file_extension == ".docx":
                # Process DOCX manually since LangChain doesn't have a built-in loader
                file_bytes = uploaded_file.read()
                document = docx.Document(BytesIO(file_bytes))
                content = "\n".join(p.text for p in document.paragraphs if p.text.strip())
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": uploaded_file.name,
                        "file_type": "docx"
                    }
                )
                documents.append(doc)
                
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            continue
    
    # Split documents into chunks
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        return split_docs
    
    return []

# --------------------------- Vector Store Setup ---------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Get HuggingFace embeddings model"""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def create_vector_store(documents: List[Document], embeddings):
    """Create FAISS vector store from documents"""
    if not documents:
        raise ValueError("No documents to create vector store")
    
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# --------------------------- LLM Setup ---------------------------
def get_llm(model_name: str, hf_token: str, max_new_tokens: int = 300, temperature: float = 0.7):
    """Get HuggingFace LLM"""
    return HuggingFaceHub(
        repo_id=model_name,
        huggingfacehub_api_token=hf_token,
        model_kwargs={
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False
        }
    )

# --------------------------- Custom Prompt Template ---------------------------
def get_custom_prompt():
    """Create custom prompt template for RAG"""
    template = """You are a helpful assistant that answers questions based on the provided context from uploaded documents.

Use ONLY the information from the provided context to answer questions. If you cannot find the answer in the context, clearly state that you don't know.

Context from documents:
{context}

Chat History:
{chat_history}

Human Question: {question}

Please provide a helpful and accurate answer based on the context above. If the context doesn't contain relevant information, say "I cannot find information about this in the uploaded documents."

Answer:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"]
    )

# --------------------------- RAG Chain Setup ---------------------------
def create_rag_chain(vector_store, llm, custom_prompt, top_k: int = 3):
    """Create conversational retrieval chain"""
    # Memory to keep track of conversation
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        k=5  # Keep last 5 exchanges
    )
    
    # Create retrieval chain
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        ),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=True,
        verbose=False
    )
    
    return retrieval_chain

# --------------------------- Session State ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "documents_ready" not in st.session_state:
    st.session_state.documents_ready = False
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    model_name = st.selectbox(
        "Chat model", 
        [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/DialoGPT-medium",
            "google/flan-t5-large",
            "HuggingFaceH4/zephyr-7b-beta"
        ], 
        index=0
    )
    
    st.subheader("Text Chunking")
    chunk_size = st.slider("Chunk size (characters)", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 50, 500, 200, 50)
    
    st.subheader("Retrieval & Generation")
    top_k = st.slider("Top-K retrieved chunks", 1, 10, 3, 1)
    max_new_tokens = st.slider("Max new tokens", 64, 1024, 300, 16)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    
    st.subheader("Embeddings Model")
    embedding_model = st.selectbox(
        "Embedding model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        ],
        index=0
    )

    show_sources = st.checkbox("Show retrieved chunks", value=True)

    if st.button("🧹 Reset chat"):
        st.session_state.chat_history = []
        if st.session_state.rag_chain and hasattr(st.session_state.rag_chain, 'memory'):
            st.session_state.rag_chain.memory.clear()
        st.success("Chat cleared.")

    if st.button("🗂️ Clear documents"):
        st.session_state.documents_ready = False
        st.session_state.rag_chain = None
        st.session_state.vector_store = None
        st.session_state.processed_files = []
        st.session_state.chat_history = []
        st.success("Documents cleared.")

    if st.session_state.chat_history:
        chat_transcript = "\n".join([
            f"{m['role'].capitalize()}: {m['content']}" 
            for m in st.session_state.chat_history
        ])
        
        if st.download_button(
            "💾 Download transcript", 
            data=chat_transcript,
            file_name="chat_transcript.txt",
            mime="text/plain"
        ):
            st.success("Transcript downloaded.")

# --------------------------- Upload & Process Documents ---------------------------
st.header("📁 Document Upload")

# Show currently processed files
if st.session_state.processed_files:
    with st.expander("📋 Currently Loaded Documents"):
        for file_info in st.session_state.processed_files:
            st.write(f"- **{file_info['name']}** ({file_info['type']}) - {file_info['chunks']} chunks")

uploaded_files = st.file_uploader(
    "Upload one or more documents (PDF / DOCX / TXT)", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True,
    help="Supported formats: PDF, DOCX, TXT. You can upload multiple files at once."
)

if uploaded_files:
    # Check if we need to reprocess (new files or different settings)
    current_file_names = [f.name for f in uploaded_files]
    processed_file_names = [f['name'] for f in st.session_state.processed_files]
    
    needs_processing = (
        set(current_file_names) != set(processed_file_names) or
        not st.session_state.documents_ready
    )
    
    if needs_processing:
        with st.spinner("Processing documents..."):
            try:
                # Process documents
                documents = process_uploaded_files(uploaded_files, chunk_size, chunk_overlap)
                
                if documents:
                    # Create embeddings
                    embeddings = get_embeddings(embedding_model)
                    
                    # Create vector store
                    vector_store = create_vector_store(documents, embeddings)
                    st.session_state.vector_store = vector_store
                    
                    # Create LLM and RAG chain
                    if hf_token:
                        llm = get_llm(model_name, hf_token, max_new_tokens, temperature)
                        
                        # Create custom prompt
                        custom_prompt = get_custom_prompt()
                        
                        # Create RAG chain
                        rag_chain = create_rag_chain(vector_store, llm, custom_prompt, top_k)
                        st.session_state.rag_chain = rag_chain
                        st.session_state.documents_ready = True
                        
                        # Update processed files info
                        st.session_state.processed_files = []
                        file_chunks = {}
                        
                        for doc in documents:
                            source = doc.metadata.get("source", "Unknown")
                            file_type = doc.metadata.get("file_type", "unknown")
                            
                            if source not in file_chunks:
                                file_chunks[source] = {"type": file_type, "count": 0}
                            file_chunks[source]["count"] += 1
                        
                        for source, info in file_chunks.items():
                            st.session_state.processed_files.append({
                                "name": source,
                                "type": info["type"].upper(),
                                "chunks": info["count"]
                            })
                        
                        st.success(f"✅ Successfully processed {len(documents)} chunks from {len(uploaded_files)} documents.")
                        
                        # Show document statistics
                        with st.expander("📊 Document Statistics"):
                            file_types = {}
                            total_chars = 0
                            
                            for doc in documents:
                                file_type = doc.metadata.get("file_type", "unknown")
                                file_types[file_type] = file_types.get(file_type, 0) + 1
                                total_chars += len(doc.page_content)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Documents", len(uploaded_files))
                                st.metric("Total Chunks", len(documents))
                            
                            with col2:
                                st.metric("Average Chunk Length", f"{total_chars // len(documents)} chars")
                                st.metric("Total Characters", f"{total_chars:,}")
                            
                            with col3:
                                st.write("**File Types:**")
                                for file_type, count in file_types.items():
                                    st.write(f"- {file_type.upper()}: {count}")
                    
                    else:
                        st.error("❌ Hugging Face token not found. Please add HUGGINGFACEHUB_API_TOKEN to your Streamlit secrets.")
                else:
                    st.warning("⚠️ No content could be extracted from the uploaded files.")
                    
            except Exception as e:
                st.error(f"❌ Error processing documents: {e}")
                st.error("Please check your files and try again.")

# --------------------------- Chat Interface ---------------------------
st.header("💬 Chat with Your Documents")

if not st.session_state.documents_ready:
    st.info("👆 Please upload documents above to start chatting.")
elif not hf_token:
    st.error("❌ Hugging Face token not found. Please add HUGGINGFACEHUB_API_TOKEN to your Streamlit secrets.")
else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if user_question := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    # Get response from RAG chain
                    result = st.session_state.rag_chain({
                        "question": user_question,
                        "chat_history": []  # LangChain memory handles this internally
                    })
                    
                    response = result["answer"]
                    source_documents = result.get("source_documents", [])
                    
                    st.write(response)
                    
                    # Show source documents if enabled
                    if show_sources and source_documents:
                        with st.expander(f"🔎 Retrieved Sources ({len(source_documents)} documents)"):
                            for i, doc in enumerate(source_documents, 1):
                                source = doc.metadata.get("source", "Unknown")
                                page = doc.metadata.get("page", "N/A")
                                file_type = doc.metadata.get("file_type", "unknown")
                                
                                # Truncate content for display
                                content = doc.page_content
                                if len(content) > 500:
                                    content = content[:500] + "..."
                                
                                st.markdown(f"**{i}. {source}** ({file_type.upper()})")
                                if page != "N/A":
                                    st.markdown(f"*Page: {page}*")
                                st.markdown(f"```\n{content}\n```")
                                
                                if i < len(source_documents):
                                    st.markdown("---")
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_message = f"❌ Error generating response: {str(e)}"
                    st.error(error_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})

# --------------------------- Footer & Configuration Display ---------------------------
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**Built with LangChain 🦜🔗 and Streamlit**")
    
    if st.session_state.documents_ready:
        st.markdown(f"📊 **Status:** {len(st.session_state.processed_files)} documents loaded, ready to chat!")
    else:
        st.markdown("📊 **Status:** No documents loaded")

with col2:
    if st.session_state.documents_ready:
        if st.button("ℹ️ Show Configuration"):
            st.info("Check the expandable section below for current settings.")

# Display current configuration
if st.session_state.documents_ready:
    with st.expander("🔧 Current Configuration"):
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.write("**Model Settings:**")
            st.write(f"- Chat Model: `{model_name}`")
            st.write(f"- Embedding Model: `{embedding_model}`")
            st.write(f"- Max Tokens: {max_new_tokens}")
            st.write(f"- Temperature: {temperature}")
        
        with config_col2:
            st.write("**Processing Settings:**")
            st.write(f"- Chunk Size: {chunk_size} characters")
            st.write(f"- Chunk Overlap: {chunk_overlap} characters")
            st.write(f"- Top-K Retrieval: {top_k} documents")
            st.write(f"- Show Sources: {'✅' if show_sources else '❌'}")

# --------------------------- Help & Instructions ---------------------------
with st.expander("❓ How to Use"):
    st.markdown("""
    ### Getting Started:
    1. **Add your Hugging Face Token** to Streamlit secrets as `HUGGINGFACEHUB_API_TOKEN`
    2. **Upload Documents** using the file uploader above (PDF, DOCX, or TXT)
    3. **Wait for Processing** - documents will be chunked and indexed
    4. **Start Chatting** - ask questions about your documents!
    
    ### Tips:
    - Upload multiple documents for broader knowledge base
    - Adjust chunk size for better context (larger = more context, smaller = more precise)
    - Use "Show retrieved chunks" to see which parts of documents are being used
    - Try different embedding models for better semantic understanding
    
    ### Troubleshooting:
    - If responses are generic, try uploading more relevant documents
    - If processing fails, check file formats and try smaller files
    - For better responses, ask specific questions about document content
    """)
