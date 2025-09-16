
import streamlit as st
from rag_pipeline import answer_query, retrieve_docs, llm_model
from vector_database import upload_pdf, build_faiss_index


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import re 


st.set_page_config(page_title="AI RAG Chatbot", layout="wide")

# #Step1: Setup Upload PDF functionality
if "history" not in st.session_state:
    st.session_state["history"] = []

st.title(" AI Chatbot with RAG")

uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    file_paths = [upload_pdf(f) for f in uploaded_files]
    st.success(f"Uploaded {len(file_paths)} PDFs")
    build_faiss_index(file_paths)
    
    
#Step2: Chatbot Skeleton (Question & Answer)

user_query = st.text_area("Ask a legal question:", height=150, placeholder="E.g. Ask Your Question Regarding the PDF")


# this is for 
col1, col2, col3 = st.columns(3)
ask_question = col1.button("Ask AI Chatbot")
summarize_pdfs = col2.button("Summarize PDFs")
clear_history = col3.button(" Clear Chat History")

if clear_history:
    st.session_state["history"] = []
    st.success("Chat history cleared!")


if ask_question and user_query.strip():
    retrieved_docs = retrieve_docs(user_query)
    answer, sources = answer_query(retrieved_docs, llm_model, user_query)

    st.session_state["history"].append({"question": user_query, "answer": answer, "sources": sources})

# Summarization feature
if summarize_pdfs:
    summary_query = "Summarize the document(s) concisely."
    retrieved_docs = retrieve_docs(summary_query, k=10)
    answer, sources = answer_query(retrieved_docs, llm_model, summary_query)

    st.session_state["history"].append({"question": "Summary of PDFs", "answer": answer, "sources": sources})

# Display chat history
for chat in st.session_state["history"]:
    st.chat_message("user").write(chat["question"])
    st.chat_message("assistant").write(chat["answer"])
    if chat["sources"]:
        st.caption("📄 Sources: " + ", ".join(chat["sources"]))

#clean the text for pdf


import re

def clean_text(text: str) -> str:
    # Remove <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove black squares
    text = text.replace("■", "")
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def generate_chat_pdf(history):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "AI Chatbot Conversation History")
    y -= 40
    c.setFont("Helvetica", 11)

    for chat in history:
        question = f"User: {clean_text(chat['question'])}"
        answer   = f"AI : {clean_text(chat['answer'])}"

        for line in [question, answer]:
            # word wrapping
            words = line.split()
            wrapped, current = [], ""
            for word in words:
                if len(current) + len(word) + 1 > 90:
                    wrapped.append(current)
                    current = word
                else:
                    current += (" " if current else "") + word
            if current:
                wrapped.append(current)

            for w in wrapped:
                if y < 50:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 11)
                c.drawString(50, y, w)
                y -= 15

        y -= 30

    c.save()
    buffer.seek(0)
    return buffer


# 📥 Download chat as PDF
if st.session_state["history"]:
    pdf_buffer = generate_chat_pdf(st.session_state["history"])
    st.download_button(
        label=" Download Chat as PDF",
        data=pdf_buffer,
        file_name="chat_history.pdf",
        mime="application/pdf",
    )
