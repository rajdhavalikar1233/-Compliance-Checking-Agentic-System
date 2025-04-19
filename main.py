import os
from rag.laws_store import build_regulation_vectorstore
from utils.pdf_reader import read_pdf

# Define paths to your PDF files
gdpr_pdf_path = r"D:\ML projects\compliance_checker2\dataset\gdpr.pdf"
ccpa_pdf_path = r"D:\ML projects\compliance_checker2\dataset\ccpa.pdf"

def main():
    print("[INFO] Reading GDPR and CCPA PDFs...")

    # Read and extract text from PDFs
    gdpr_text = read_pdf(gdpr_pdf_path)
    ccpa_text = read_pdf(ccpa_pdf_path)

    if not gdpr_text or not ccpa_text:
        raise RuntimeError("PDF reading failed for one or both files.")

    # Construct the regulation dictionary
    regulations = {
        "GDPR": gdpr_text,
        "CCPA": ccpa_text
    }

    # Build and store vector databases
    print("[INFO] Creating and saving regulation vector stores...")
    build_regulation_vectorstore(regulations)

    print("[SUCCESS] Regulation vector stores created successfully.")


if __name__ == "__main__":
    main()
