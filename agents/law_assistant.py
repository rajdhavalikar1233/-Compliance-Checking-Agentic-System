import streamlit as st
from rag.laws_store import build_regulation_vectorstore, load_regulation_vectorstore
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load env variables (TOGETHER_API_KEY)
load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")

# --- STEP 1: Build vectorstore ---
def prepare_vectorstores():
    with open("docs/gdpr.txt", "r", encoding="utf-8") as f:
        gdpr_text = f.read()
    with open("docs/ccpa.txt", "r", encoding="utf-8") as f:
        ccpa_text = f.read()

    regulation_docs = {
        "GDPR": gdpr_text,
        "CCPA": ccpa_text
    }

    build_regulation_vectorstore(regulation_docs)

# --- STEP 2: Load QA chain ---
def create_qa_chain(law_name):
    vectorstore = load_regulation_vectorstore(law_name)
    llm = ChatOpenAI(
        temperature=0,
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        openai_api_key=together_api_key,
        base_url="https://api.together.xyz/v1"
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain