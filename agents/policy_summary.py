import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# Load API key
groq_api_key = os.getenv("GROQ_API_KEY")

# ---------------------- Load the LLM ----------------------
def load_llm():
    return ChatGroq(
        temperature=0.4,
        model_name="llama3-8b-8192",
        api_key=groq_api_key
    )

# ---------------- Load FAISS Vector Store ----------------
def load_vector_store(path):
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(folder_path=path, embeddings=embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load vector store: {e}")

# ----------- Pretty Print the Summary Output ------------
def print_clean_summary(output):
    print("\n--- 📄 POLICY SUMMARY ---\n")

    # Attempt to parse sections from the response
    try:
        # Separate JSON, summary, and risk analysis
        json_start = output.find("{")
        json_end = output.find("}\n", json_start) + 1
        json_text = output[json_start:json_end]
        parsed_json = json.loads(json_text)

        # Print parsed JSON in clean format
        for section, items in parsed_json.items():
            print(f"\033[1m{section}:\033[0m")  # bold
            for item in items:
                for key, value in item.items():
                    print(f"  - {key}: {value}")
            print()

        # Extract and print descriptive summary
        if "Descriptive Natural Language Summary:" in output:
            nat_summary = output.split("Descriptive Natural Language Summary:")[1].split("**Potential Risks")[0].strip()
            print("\n--- ✍️ NATURAL LANGUAGE SUMMARY ---\n")
            print(nat_summary)

        # Extract and print risk points
        if "Potential Risks or Non-Compliance:" in output:
            risks_text = output.split("Potential Risks or Non-Compliance:")[1].strip()
            risks = risks_text.split("\n")
            print("\n--- ⚠️ POTENTIAL RISKS OR NON-COMPLIANCE ---\n")
            for line in risks:
                if line.strip():
                    print(line.strip())

    except Exception as e:
        print("[WARNING] Failed to cleanly parse summary output. Showing raw text:\n")
        print(output)

# ------------- Query & Summarize Policies ---------------
def query_policy_summary(vectorstore, query, k=6):
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""
You are a legal language expert and policy summarization assistant.

Given the policy text below, write a professional, easy-to-read summary that includes the following sections:

1. *Overview* – Briefly describe what the policy is about.
2. *Data Collection* – What user data is collected and how.
3. *User Rights* – Mention rights like access, correction, deletion, etc.
4. *Consent & Preferences* – How user consent is taken and managed.
5. *Third-party Sharing* – Any mention of data shared with other entities.
6. *Data Retention & Security* – How long data is kept and security measures.
7. *Risks or Compliance Issues* – If there are any risks, inconsistencies, or non-compliance with privacy regulations like GDPR, CCPA, etc., list them as bullet points.

- Keep it in clear, natural language.
- Do *not* return JSON or structured data.
- Avoid repetition and ensure it’s readable for compliance/legal reviewers.

Policy Text:
{context}
"""

    llm = load_llm()
    response = llm.invoke(system_prompt)
    return response.content

# --------------- Wrapper for full flow -------------------
def run_policy_summary_retrieval(vectorstore, query_keywords):
    try:
        print("[INFO] Querying policy summaries...")
        return query_policy_summary(
            vectorstore=vectorstore,
            query=query_keywords
        )
    except Exception as err:
        raise RuntimeError(f"[POLICY EXTRACTOR ERROR] {err}")

# -------------------- Main Block -------------------------
if __name__ == "__main__":
    vectorstore_path = r"D:\ML projects\compliance_checker 3\data\vector_stores\policy_store"

    print("[INFO] Loading vector store from disk...")
    store = load_vector_store(vectorstore_path)

    query_keywords = "data collection, user rights, consent, third-party sharing, retention periods, privacy compliance"

    print("[INFO] Running policy summarization...")
    summary = run_policy_summary_retrieval(store, query_keywords)

    print_clean_summary(summary)
