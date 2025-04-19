import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Query mapping for each type of policy
query_map = {
    "Privacy Policy": [
        "How is personal data collected?",
        "What third-party services are used?",
        "What personal information is shared with others?",
        "User consent mechanism and cookies?",
        "How can a user delete their data?",
        "How is user tracking handled?"
    ],
    "Terms of Use": [
        "Liability disclaimers and limitations",
        "Data ownership clauses",
        "Jurisdiction and dispute resolution",
        "Account and password responsibility"
    ],
    "Cookie Policy": [
        "Types of cookies used",
        "Purpose of cookies",
        "Cookie control and settings",
        "Consent for tracking cookies"
    ]
}

# Load vector DB for GDPR or CCPA
def load_vector_db(regulation: str) -> FAISS:
    if regulation.lower() not in ["gdpr", "ccpa"]:
        raise ValueError("Regulation must be 'gdpr' or 'ccpa'")

    db_path = f"data/vector_stores/regulations/{regulation.lower()}"
    if not os.path.exists(os.path.join(db_path, "index.faiss")):
        raise FileNotFoundError(f"Vector DB not found at {db_path}. Run main.py to build it.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

# Fetch relevant docs for a specific policy type (used in run_agents)
def get_relevant_docs(policy_type: str, regulation: str = "gdpr", top_k: int = 5) -> list[str]:
    queries = query_map.get(policy_type)
    if not queries:
        raise ValueError(f"Unknown policy type: {policy_type}")

    db = load_vector_db(regulation)

    retrieved_docs = []
    for query in queries:
        docs: list[Document] = db.similarity_search(query, k=top_k)
        for doc in docs:
            retrieved_docs.append(doc.page_content)

    # Deduplicate and limit length for context prompt
    unique_context = list(dict.fromkeys(retrieved_docs))
    return unique_context

import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # remove excessive whitespace
    return text.strip()

def truncate_text(text, max_tokens=3000):
    words = text.split()
    return ' '.join(words[:max_tokens]) if len(words) > max_tokens else text
