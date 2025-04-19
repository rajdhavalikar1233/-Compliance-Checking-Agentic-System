import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.utils import clean_text

# Load the embedding model (can be reused across pipelines)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def chunk_regulation_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


def build_regulation_vectorstore(regulation_docs: dict, base_save_path="./data/vector_stores/regulations"):
    """
    Builds a FAISS vectorstore for each regulation (GDPR, CCPA).
    
    Args:
        regulation_docs: dict with keys as regulation names (e.g., "GDPR") and values as full regulation text.
        base_save_path: base directory where vector DBs will be saved separately per regulation.
    
    Returns:
        dict of regulation name to their FAISS vector store objects.
    """
    vectorstores = {}

    for law_name, full_text in regulation_docs.items():
        print(f"[INFO] Processing regulation: {law_name}")
        cleaned_text = clean_text(full_text)
        chunks = chunk_regulation_text(cleaned_text)
        metadatas = [{"law": law_name} for _ in chunks]

        vectorstore = FAISS.from_texts(chunks, embedding=embedding_model, metadatas=metadatas)
        
        save_path = os.path.join(base_save_path, law_name.lower().replace(" ", "_"))
        vectorstore.save_local(save_path)
        vectorstores[law_name] = vectorstore

        print(f"[INFO] Saved {law_name} vector store at: {save_path}")

    return vectorstores


def load_regulation_vectorstore(law_name, base_path="./data/vector_stores/regulations"):
    """
    Loads a saved regulation vector store (GDPR or CCPA).
    """
    load_path = os.path.join(base_path, law_name.lower().replace(" ", "_"))
    return FAISS.load_local(load_path, embeddings=embedding_model,allow_dangerous_deserialization=True)
