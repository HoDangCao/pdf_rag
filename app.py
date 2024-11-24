import streamlit as st
import fitz
import pytesseract
from PIL import Image
import io

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def pdf_read(pdf_doc):
    text = ''
    for pdf in pdf_doc:
        if type(pdf) == str: # read with path
            pdf_reader = fitz.open(pdf)
        else: # read with file object
            pdf_reader = fitz.open(stream=pdf.read())
            
        for page in pdf_reader:         
            text_page = page.get_text()

            if not text_page.strip(): # if no text => scanned file
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes('png')))
                text_page = pytesseract.image_to_string(img, config=st.session_state.ocr_config)

            text += text_page
    return text.strip()

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    return text_splitter.split_text(text)

def vector_store(text_chunks):    
    vector_store = FAISS.from_texts(text_chunks, embedding=st.session_state.embeddings)
    vector_store.save_local("faiss_db")

def get_context(user_question):
    new_db = FAISS.load_local('faiss_db', st.session_state.embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs={"k": 3}) # return the top_k chunk.
    retriever_chain = create_retriever_tool(retriever, 'pdf_extractor', 'This tool is to give answer to queries from the pdf')
    return retriever_chain.invoke({'query': user_question})

def get_response(history):
    prompt = ChatPromptTemplate.from_messages(history)
    chain = prompt | st.session_state.llm | StrOutputParser()
    return chain.invoke({}, temperature=0).strip()

def main():
    st.set_page_config("Chat PDF")
    st.header("RAG based Chat with PDF")

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
    if "llm" not in st.session_state:
        st.session_state.llm = ChatOllama(model='mistral', quantization='4bit')
    if "ocr_config" not in st.session_state:
        st.session_state.ocr_config = r'--psm 11 --oem 3'
    if "history" not in st.session_state:
        st.session_state.history = [
            AIMessage(content="Hello! I'm an AI assistant that gives answer about the pdf files for your reasonable question"),
        ]

    for message in st.session_state.history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("User"):
                st.markdown(message.content)

    user_question = st.chat_input("Ask a Question from the PDF Files")

    if user_question and user_question.strip() != "":
        with st.chat_message("User"):
            st.markdown(user_question)
        
        with st.chat_message("AI"):
            related_chunk = get_context(user_question)            
            st.session_state.history.append(("user", f"Given document: {related_chunk}"))
            st.session_state.history.append(HumanMessage(content=user_question))
            response = get_response(st.session_state.history)
            st.markdown(response)
            st.session_state.history.append(AIMessage(content=response))

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", type=['pdf'], accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if not pdf_docs:
                    st.error("Please upload at least one PDF file!")
                else:
                    raw_text = pdf_read(pdf_docs)
                    if not raw_text:
                        st.error("Could not read the file. Please upload a different file.")
                    else:
                        text_chunks = get_chunks(raw_text)
                        vector_store(text_chunks)
                        st.success("Processing complete!")

if __name__ == "__main__":
    main()