import streamlit as st
import fitz
import pytesseract
from PIL import Image
import io

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.llms import HuggingFacePipeline
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
ocr_config = r' --psm 11 --oem 3'

def pdf_read(pdf_doc):
    text = ''
    for pdf in pdf_doc:
        if type(pdf) == str: # read with path
            pdf_reader = fitz.open(pdf)
        else: # read with file object
            pdf_reader = fitz.open(stream=pdf.read())
            
        for page_num in range(pdf_reader.page_count):
            page = pdf_reader[page_num]            
            text_page = page.get_text()

            if not text_page.strip(): # if no text => scanned file
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes('png')))
                text_page = pytesseract.image_to_string(img, config=ocr_config)

            text += text_page
    return text.strip()

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

def user_input(user_question):
    new_db = FAISS.load_local('faiss_db', embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs={"k": 1}) # top_k=1 will return only the top chunk.
    retriever_chain = create_retriever_tool(retriever, 'pdf_extractor', 'This tool is to give answer to queries from the pdf')
    response = get_conversational_chain(retriever_chain, user_question)
    return response

def get_conversational_chain(tool, ques):
    # Load the model and tokenizer locally from Hugging Face
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

    # Create the Hugging Face pipeline for conversational tasks with specified max_length
    generation_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        device='cuda' if torch.cuda.is_available() else 'cpu',
        # max_length=1024,
        max_new_tokens=200, 
        pad_token_id=50256
    )

    # Wrap the pipeline with HuggingFacePipeline to integrate with LangChain
    llm = HuggingFacePipeline(pipeline=generation_pipeline)

    # Define your prompt template that includes the 'agent_scratchpad' variable
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant with access to the following tool: {tool_name}. {tool_description}"),
        ("user", "Given document: {related_chunk}"), 
        ("user", "{query}"),
        ("assistant", "Let me think...")
    ])

    # Manually set up an agent executor that can use the LLM and tools
    def custom_agent_executor(user_input):
        # Call the tool to get the response
        tool_response = tool.invoke({'query': user_input})  # Provide 'query' instead of 'input'
        
        # Print the tool response to inspect its structure
        print("Tool Response:", tool_response)

        # Check if tool_response is a string or dict and access accordingly
        if isinstance(tool_response, dict):
            agent_scratchpad = tool_response.get('output', "No output found.")
        else:
            agent_scratchpad = tool_response

        # Combine tool response with user input and prompt
        formatted_prompt = prompt.format(
            tool_name=tool.name,
            tool_description=tool.description,
            related_chunk=agent_scratchpad,
            query=user_input,
        )

        # Call the LLM with the formatted prompt
        response = llm.invoke(formatted_prompt, temperature=0)

        # Handle the response correctly
        if isinstance(response, list) and len(response) > 0:
            return response[0][len(formatted_prompt):].strip()  # If response is a list, return the first element directly
        elif isinstance(response, str):
            return response[len(formatted_prompt):].strip()  # If response is a string, return it directly
        else:
            return "No valid response received."  # Handle unexpected response formats

    # Execute the custom agent executor
    response = custom_agent_executor(ques)

    # Output the response
    st.write('reply: ', response)
    
def main():
    st.set_page_config("Chat PDF")
    st.header("RAG based Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        try:
            user_input(user_question)
        except RuntimeError as e:
            st.error(f'Some error happend {e}')

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", type=['pdf'], accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                if raw_text == '':
                    st.error('can NOT read this file. Please upload another files!!')
                else:
                    text_chunks = get_chunks(raw_text)
                    vector_store(text_chunks)
                    st.success("Done")

if __name__ == "__main__":
    main()