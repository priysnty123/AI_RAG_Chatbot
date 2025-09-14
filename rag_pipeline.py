
from langchain_groq import ChatGroq
from vector_database import load_faiss_index
from langchain_core.prompts import ChatPromptTemplate

#step 1 llm setup 
llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")


#Step2: Retrieve Docs

def retrieve_docs(query, k=3):
    faiss_db = load_faiss_index()
    return faiss_db.similarity_search(query, k=k)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context


#Step3: Answer Question
custom_prompt_template = """
Use ONLY the provided context to answer the question.
If unsure, say "I don’t know".
Do not include reasoning steps or <think> tags.
Question: {question}
Context: {context}
Answer:
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    response = chain.invoke({"question": query, "context": context})

    # Collect sources
    sources = []
    for doc in documents:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        sources.append(f"{src} (Page {page})")

    return response.content.strip(), list(set(sources))
