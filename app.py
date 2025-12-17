import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

hf_api_key = os.getenv('HF_TOKEN')
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("ASK YOUR DOCS 📃🗣️")

# Session STATE
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None
    
if 'retriever' not in st.session_state:
    st.session_state.retriever = None


uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and uploaded_file.name != st.session_state.file_name:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever()

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        api_key = groq_api_key
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a factual assistant.
        Answer the question using ONLY the provided context.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question:   
        {question}

        Answer:
        """
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    st.session_state.rag_chain = rag_chain
    st.session_state.file_name = uploaded_file.name
    st.session_state.retriever = retriever

    st.success("PDF indexed successfully!")


if st.session_state.rag_chain:
    question = st.text_input("Ask your question")

    if question:
        answer = st.session_state.rag_chain.invoke(question)
        st.write(answer)
