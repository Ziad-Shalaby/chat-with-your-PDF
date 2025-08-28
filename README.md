# Chat with Your PDF 📄💬

## Overview
Chat with Your PDF is a Streamlit-based web application that allows you to upload PDF files and interact with them using natural language. Powered by **Mistral-7B-Instruct** through the Hugging Face Inference API, this app extracts text from your PDFs and enables question-answering or conversational interaction with the content.

## Live Demo
Access the live application here: [Chat with Your PDF](https://chat-with-your-pdf-1.streamlit.app/)

## Features
- Upload and process any PDF file.
- Extract and embed text using sentence transformers.
- Ask questions and get instant answers about your PDF content.
- User-friendly Streamlit interface.
- Secure token management using Streamlit Secrets.

## Tech Stack
- **Python 3.10+**
- **Streamlit**
- **Hugging Face Inference API**
- **FAISS** for vector search
- **Sentence Transformers** for text embeddings

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/chat-with-your-pdf.git
cd chat-with-your-pdf
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Add your Hugging Face API token to `.streamlit/secrets.toml`:
```toml
HUGGINGFACEHUB_API_TOKEN = "your_huggingface_api_token_here"
```
4. Run the app:
```bash
streamlit run app.py
```

## Usage
1. Launch the app using the command above.
2. Upload your PDF file.
3. Ask questions in the chat interface.
4. Get answers instantly!

## Project Structure
```
chat-with-your-pdf/
│
├── app.py                 # Main application file
├── requirements.txt       # Required dependencies
└── .streamlit/
    └── secrets.toml       # API token storage
```

## Contributing
Contributions are welcome! Feel free to fork the repository, submit pull requests, or report issues.

## Author
Created by **Ziad Shalaby**

