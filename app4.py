import streamlit as st
import os
from dotenv import load_dotenv

# Local modules
from webscraper import main as scrape_main
from rag.policy_store import build_policy_vectorstore, save_vectorstore
from rag.laws_store import build_regulation_vectorstore, load_regulation_vectorstore
from agents.policy_summary import run_policy_summary_retrieval, load_vector_store
from utils.query_map import query_map
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import json
import warnings
from pydantic import PydanticDeprecationWarning

warnings.filterwarnings("ignore", category=PydanticDeprecationWarning)

# Load API key
load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")

llm = ChatOpenAI(
    temperature=0,
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    openai_api_key=together_api_key,
    base_url="https://api.together.xyz/v1"
)

# Load Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Load the LLM ---
def load_llm():
    return ChatGroq(
        temperature=0.4,
        model_name="llama3-8b-8192",
        api_key=groq_api_key
    )

# ----------------------- Regulation Chatbot Setup -----------------------
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

# --- Compliance Check Prompt Template ---
def generate_compliance_prompt(policy_text, gdpr_text, ccpa_text, questions, policy_type):
    prompt = f"""
You are a Privacy Compliance Expert. You need to check if the given policy is compliant with {policy_type}.

Below are relevant reference documents (from regulations like GDPR/CCPA):
---------------------
Policy:
{policy_text}
---------------------
GDPR:
{gdpr_text}
---------------------
CCPA:
{ccpa_text}
---------------------

Based on these references, analyze the provided policy and determine whether it is compliant.

Please answer in the following STRICT JSON format only (without any markdown, explanation outside the JSON, or commentary):

Your Response Format (Strict JSON):
{{
    "overall_status": "Compliant/Non-Compliant/Partially Compliant",
    "explanation": "A high-level summary explaining your decision",
    "questions": [
        {{
            "question": "Does the policy mention how user data is collected?",
            "status": "Yes/No/Partially",
            "explanation": "Explain why based on the policy content",
            "regulation": "GDPR Article 5"  // or relevant CCPA section
        }},
        {{
            "question": "Does the policy mention the user’s right to delete their data?",
            "status": "...",
            "explanation": "...",
            "regulation": "..."
        }}
        // Add 3-5 key questions based on compliance requirements
    ]
}}
Only output the JSON.
"""
    return prompt.strip()


def display_compliance_report(report):
    st.markdown(f"### 📋 Overall Status: {report['overall_status']}")
    st.markdown(f"Explanation: {report['explanation']}")

    st.markdown("---")
    st.markdown("### 🧩 Question-wise Breakdown:")

    for q in report["questions"]:
        st.markdown(f"Q: {q['question']}")
        st.markdown(f"- Status: {q['status']}")
        st.markdown(f"- Explanation: {q['explanation']}")
        st.markdown(f"- Regulation: {q['regulation']}")
        st.markdown("---")

# ----------------------- Streamlit UI -----------------------
st.set_page_config(page_title="🛡 Privacy Policy Compliance Agent", layout="wide")
st.title("🛡 Privacy Policy Compliance Agentic System")

# Global constants
policy_vectorstore_path = "data/vector_stores/policy_store"
regulations = ["GDPR", "CCPA"]

# Tabs
tab1, tab2, tab3 ,tab4 = st.tabs(["🔍 Scrape & Store", "📝 Policy Summary", "📘 Regulation Q&A", "Compliance Analyst"])

# ----------------------- TAB 1: Scrape & Store -----------------------
with tab1:
    st.header("🔍 Scrape Privacy Policy & Create Vector Store")
    website_url = st.text_input("Enter Website URL", placeholder="https://www.amazon.in")

    if st.button("Scrape and Store"):
        if not website_url:
            st.error("Please enter a valid website URL.")
        else:
            with st.spinner("Scraping policy..."):
                scraped_path = scrape_main(website_url)
            st.success("✅ Policy scraped successfully!")

            with open(scraped_path, "r", encoding="utf-8") as f:
                policy_text = f.read()

            st.info("Building vectorstore from policy...")
            vectorstore = build_policy_vectorstore(policy_text, website_url)
            save_vectorstore(vectorstore)
            st.success("✅ Vector store created and saved.")
            st.subheader("📄 Sample Policy Preview")
            st.code(policy_text[:1500])

# ----------------------- TAB 2: Policy Summary -----------------------
with tab2:
    st.header("📝 Generate Policy Summary")

    if st.button("Generate Summary"):
        try:
            with st.spinner("Generating summary..."):
                vectorstore = load_vector_store(policy_vectorstore_path)
                query = "data collection, user rights, consent, third-party sharing, retention periods, privacy compliance"
                summary = run_policy_summary_retrieval(vectorstore, query)
            st.success("✅ Summary Ready")
            st.subheader("📄 Policy Summary")
            st.text_area("Generated Summary", summary, height=400)
        except Exception as e:
            st.error(f"❌ Summary failed: {e}")

# ----------------------- TAB 3: Regulation Chatbot -----------------------
with tab3:
    st.header("📘 Regulation Q&A Chatbot: GDPR & CCPA")

    if st.button("🔄 Rebuild Regulation Vectorstores"):
        with st.spinner("Rebuilding vectorstores..."):
            prepare_vectorstores()
        st.success("✅ Vectorstores rebuilt successfully!")

    selected_law = st.selectbox("Choose a regulation", regulations)

    if "qa_chain" not in st.session_state or st.session_state.get("law") != selected_law:
        st.session_state.qa_chain = create_qa_chain(selected_law)
        st.session_state.chat_history = []
        st.session_state.law = selected_law

    query = st.chat_input(f"Ask something about {selected_law}...")

    if query:
        with st.spinner("Getting answer..."):
            result = st.session_state.qa_chain.run(query)
            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("bot", result))

    for role, message in st.session_state.chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(message)

# ----------------------- TAB 4: Policy Compliance Checker -----------------------
with tab4:
    st.header("✅ Policy Compliance Checker")

    policy_type = st.selectbox("Select Policy Type", list(query_map.keys()))
    check_button = st.button("Check Compliance")

    if check_button:
        try:
            with st.spinner("🔄 Loading vectorstores..."):
                # Load vector stores for policy, GDPR, and CCPA
                policy_vs = load_vector_store("data/vector_stores/policy_store")
                gdpr_vs = load_regulation_vectorstore("GDPR")
                ccpa_vs = load_regulation_vectorstore("CCPA")

            # Get the related questions for the selected policy type
            questions = query_map[policy_type]

            with st.spinner("🔍 Retrieving relevant documents..."):
                # Retrieve documents for each regulation (policy, GDPR, and CCPA)
                policy_docs = policy_vs.similarity_search(" ".join(questions), k=8)
                gdpr_docs = gdpr_vs.similarity_search(" ".join(questions), k=8)
                ccpa_docs = ccpa_vs.similarity_search(" ".join(questions), k=8)

            # Combine the text from the retrieved documents
            policy_text = "\n\n".join([doc.page_content for doc in policy_docs])
            gdpr_text = "\n\n".join([doc.page_content for doc in gdpr_docs])
            ccpa_text = "\n\n".join([doc.page_content for doc in ccpa_docs])

            # Load the LLM model
            llm = load_llm()

            # Generate the compliance prompt
            full_prompt = generate_compliance_prompt(policy_text, gdpr_text, ccpa_text, questions, policy_type)

            with st.spinner("🤖 Analyzing compliance using LLM..."):
                llm_response = llm.invoke(full_prompt)

            # Extract text content from LLM response
            llm_text = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

            # Safely extract data: Try loading structured JSON from LLM response
            try:
                report_dict = json.loads(llm_text)
            except json.JSONDecodeError:
                st.error("❌ LLM response is not valid JSON. Here's the raw response:")
                st.code(llm_text)
                raise

            # Display success message and the compliance report
            st.success("✅ Compliance Check Completed")
            display_compliance_report(report_dict)

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")