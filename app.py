import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Set up LLM and retriever
llm = Ollama(model="mistral")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Custom prompt
prompt = PromptTemplate.from_template("""
You are a helpful medical assistant. Use the following information to answer 
the patient's question about their symptoms. Be clear, concise, and always 
recommend seeing a doctor for proper diagnosis.

Context from medical knowledge base:
{context}

Patient's symptoms/question:
{question}

Your response:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

DISCLAIMER = """
---
⚠️ **Medical Disclaimer:** This tool is for informational purposes only 
and is NOT a substitute for professional medical advice, diagnosis, or treatment. 
Always consult a qualified doctor.
"""

# Streamlit UI
st.set_page_config(page_title="MedRAG Symptom Checker", page_icon="🩺")
st.title("🩺 MedRAG — Symptom Checker")
st.write("Describe your symptoms below and get information from our medical knowledge base.")

symptoms = st.text_area(
    "What symptoms are you experiencing?",
    placeholder="e.g. I have had a fever, sore throat and headache for the past 2 days...",
    height=150
)

if st.button("Check Symptoms", type="primary"):
    if symptoms.strip() == "":
        st.warning("Please describe your symptoms first.")
    else:
        with st.spinner("Searching medical knowledge base..."):
            response = chain.invoke(symptoms)
        st.subheader("📋 Results")
        st.write(response)
        st.markdown(DISCLAIMER)
        