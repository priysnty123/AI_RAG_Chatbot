
 #Step 1: Upload & Load raw PDF(s)

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

import os

pdfs_directory = 'pdfs/'
FAISS_DB_PATH = "vectorstore/db_faiss"
ollama_model_name = "deepseek-r1:1.5b"

def upload_pdf(file):
    os.makedirs(pdfs_directory, exist_ok=True)
    file_path = os.path.join(pdfs_directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()


# #Step 2: Create Chunks
def create_chunks(documents, source_name):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    # Attach metadata (source + page number)
    for c in chunks:
        c.metadata["source"] = source_name
        c.metadata["page"] = c.metadata.get("page", None)
    return chunks


#Step 3: Setup Embeddings Model (Use DeepSeek R1 with Ollama)
# def get_embedding_model():
#     return OllamaEmbeddings(model=ollama_model_name)


from langchain_community.embeddings import HuggingFaceEmbeddings

# def get_embedding_model():
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
import os
from langchain_community.embeddings import HuggingFaceEmbeddings

# Try importing Ollama locally only
try:
    from langchain_ollama import OllamaEmbeddings
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

ollama_model_name = "deepseek-r1:1.5b"

def get_embedding_model():
    """Use HuggingFace on Streamlit Cloud (CPU only), Ollama locally if available."""
    running_in_streamlit_cloud = os.environ.get("STREAMLIT_SERVER_HEADLESS") == "1"

    if running_in_streamlit_cloud:
        print("🌐 Running on Streamlit Cloud → Using HuggingFace embeddings (CPU only)")
        model_kwargs = {"device": "cpu"}   # ✅ fix meta tensor error
        encode_kwargs = {"normalize_embeddings": True}
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    if OLLAMA_AVAILABLE:
        print("💻 Running locally → Using Ollama embeddings")
        return OllamaEmbeddings(model=ollama_model_name)

    print("Fallback → Using HuggingFace embeddings (CPU only)")
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

#Step 4: Index Documents **Store embeddings in FAISS (vector store)
def build_faiss_index(uploaded_files):
    all_chunks = []
    for file_path in uploaded_files:
        documents = load_pdf(file_path)
        file_name = os.path.basename(file_path)
        chunks = create_chunks(documents, source_name=file_name)
        all_chunks.extend(chunks)

    faiss_db = FAISS.from_documents(all_chunks, get_embedding_model())
    faiss_db.save_local(FAISS_DB_PATH)
    return faiss_db

def load_faiss_index():
    return FAISS.load_local(FAISS_DB_PATH, get_embedding_model(), allow_dangerous_deserialization=True)
