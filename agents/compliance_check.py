import streamlit as st
from rag.policy_store import load_policy_vector
from rag.laws_store import load_regulation_vectorstore
from query_map import query_map
from langchain.chat_models import ChatGroq
from langchain.schema import Document
import os

# Load API Key
from dotenv import load_dotenv
load_dotenv()

# Load Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Load the LLM ---
def load_llm():
    return ChatGroq(
        temperature=0.4,
        model_name="llama3-8b-8192",
        api_key=groq_api_key
    )

# --- Compliance Check Prompt Template ---
def generate_compliance_prompt(policy_docs, gdpr_docs, ccpa_docs, questions):
    prompt = f"""
You are a senior legal compliance auditor with expertise in data privacy laws like GDPR and CCPA.

You are tasked with evaluating a website's privacy-related policy documents to check whether they fully comply with GDPR and CCPA. Use the official regulation texts provided and ensure your evaluation is based only on these.

---

📄 **Website Policy Documents**:
{policy_docs}

📘 **GDPR Law Reference**:
{gdpr_docs}

📙 **CCPA Law Reference**:
{ccpa_docs}

---

🔍 **Evaluation Criteria**:
- Consider the following specific questions and criteria:
{questions}

---

📝 **Your Output Must Include**:

1. **Summary of Findings**:
    - A short executive summary explaining whether the document is largely compliant, partially compliant, or non-compliant.

2. **Detailed Section-wise Assessment**:
    For each criteria/question:
    - State whether it's:
        - ✅ Compliant
        - ⚠ Partially Compliant
        - ❌ Non-Compliant
    - Justify your evaluation with:
        - A clear explanation
        - Specific quotes or sections from the policy
        - Corresponding regulation clauses from GDPR/CCPA

3. **Recommended Actions**:
    - For each non-compliant or partially compliant item, suggest precise corrective actions.

4. **Confidence Level**:
    - Mention your confidence level in this evaluation (High / Medium / Low).

Be objective and cite sources wherever possible.
    """.strip()
    return prompt

# ---------------- TAB 4: Compliance Checker ----------------
tab4 = st.tabs(["✅ Compliance Checker"])[0]

with tab4:
    st.header("✅ Policy Compliance Checker")

    policy_type = st.selectbox("Select Policy Type", list(query_map.keys()))
    check_button = st.button("Check Compliance")

    if check_button:
        try:
            with st.spinner("Loading vectorstores..."):
                policy_vs = load_policy_vector("data/vector_stores/policy_store")
                gdpr_vs = load_regulation_vectorstore("GDPR")
                ccpa_vs = load_regulation_vectorstore("CCPA")

            questions = query_map[policy_type]

            # Fetch top 8 relevant docs for each vectorstore
            with st.spinner("Retrieving documents..."):
                policy_docs = policy_vs.similarity_search(" ".join(questions), k=8)
                gdpr_docs = gdpr_vs.similarity_search(" ".join(questions), k=8)
                ccpa_docs = ccpa_vs.similarity_search(" ".join(questions), k=8)

            # Combine docs into strings
            policy_text = "\n\n".join([doc.page_content for doc in policy_docs])
            gdpr_text = "\n\n".join([doc.page_content for doc in gdpr_docs])
            ccpa_text = "\n\n".join([doc.page_content for doc in ccpa_docs])

            # Load LLM and create prompt
            llm = load_llm()
            full_prompt = generate_compliance_prompt(policy_text, gdpr_text, ccpa_text, questions)

            with st.spinner("Analyzing compliance using LLM..."):
                response = llm.invoke(full_prompt)

            st.success("✅ Compliance Check Completed")
            st.subheader("📊 Compliance Report")
            st.text_area("Compliance Report", response, height=600)

        except Exception as e:
            st.error(f"❌ Error checking compliance: {e}")
