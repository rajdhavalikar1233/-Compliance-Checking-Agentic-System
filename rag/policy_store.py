import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from utils.utils import clean_text

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def chunk_policy_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

def build_policy_vectorstore(text, url):
    cleaned = clean_text(text)
    chunks = chunk_policy_text(cleaned)
    metadatas = [{"source_url": url} for _ in chunks]

    return FAISS.from_texts(chunks, embedding=embedding_model, metadatas=metadatas)

def save_vectorstore(vectorstore, path="./data/vector_stores/policy_store"):
    vectorstore.save_local(path)
