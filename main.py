import streamlit as st
import pdfplumber
import re
from transformers import pipeline

# ---------------------------
# 1. Extract text from PDF
# ---------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text.strip() + "\n"
    return text

# ---------------------------
# 2. Clean extracted text
# ---------------------------
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)  # Remove extra newlines
    text = re.sub(r'\s+', ' ', text)   # Normalize spaces
    return text.strip()

# ---------------------------
# 3. Split text into chunks (to avoid token limits)
# ---------------------------
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ---------------------------
# 4. Load model based on mode
# ---------------------------
@st.cache_resource
def load_model(task):
    if task == "Summary Only":
        # Smaller model, works well for summarization
        return pipeline("text2text-generation", model="google/flan-t5-base")
    else:
        # Larger model, better for complex prompts & Q&A
        return pipeline("text2text-generation", model="google/flan-t5-large")

# ---------------------------
# 5. Summarize or answer prompt
# ---------------------------
def generate_response(text, prompt, model):
    chunks = chunk_text(text)
    responses = []

    for chunk in chunks:
        input_text = f"{prompt}\n{chunk}"
        result = model(input_text, max_length=256, truncation=True)
        responses.append(result[0]["generated_text"])

    return "\n".join(responses)

# ---------------------------
# 6. Streamlit App
# ---------------------------
def main():
    st.title("📄 Advanced PDF Summarizer")
    st.markdown("Upload a PDF and either summarize it or ask a custom question.")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    mode = st.selectbox("Choose Mode", ["Summary Only", "Custom Prompt"])

    if mode == "Custom Prompt":
        user_prompt = st.text_area("Enter your question or prompt")
    else:
        user_prompt = "Summarize this document in 5 concise bullet points:"

    if uploaded_file and user_prompt:
        st.info("Extracting text from PDF...")
        pdf_text = extract_text_from_pdf(uploaded_file)
        pdf_text = clean_text(pdf_text)

        if not pdf_text.strip():
            st.error("No extractable text found. The PDF may be scanned. Try OCR-based tools.")
            return

        st.write(f"📄 Extracted {len(pdf_text.split())} words from the document.")

        # pass `mode` into load_model
        model = load_model(mode)

        with st.spinner("Generating response..."):
            final_response = generate_response(pdf_text, user_prompt, model)

        st.subheader("Response")
        st.write(final_response)

        # Option to download
        st.download_button("Download Response", final_response, file_name="summary.txt")

if __name__ == "__main__":
    main()
