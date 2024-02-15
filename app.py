import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import langchain
import pandas as pd
from langchain_community.document_loaders import UnstructuredHTMLLoader

langchain.verbose = False

load_dotenv()


def process_pdf(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base


def process_csv(text):
    # Concatenate all the text from CSV columns
    text = ' '.join(text)
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base


def main():
    st.title("Preguntas a un archivo")

    file_type = st.radio("Selecciona el tipo de archivo:", ("rss", "html"))

    file = st.file_uploader("Sube tu archivo", type=["rss", "html"])

    if file is not None:
        if file_type == "rss":
            rss_reader = PdfReader(file)
            text = ""
            for page in rss_reader.pages:
                text += page.extract_text()
            knowledge_base = process_pdf(text)
        if file_type == "html":

            uploader = UnstructuredHTMLLoader()
            data =uploader.load()
            data


        query = st.text_input('Escribe tu pregunta para el archivo...')
        cancel_button = st.button('Cancelar')

        if cancel_button:
            st.stop()

        if query:
            docs = knowledge_base.similarity_search(query)
            model = "gpt-3.5-turbo-instruct"
            temperature = 0
            llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cost:
                response = chain.invoke(input={"question": query, "input_documents": docs})
                print(cost)
                st.write(response["output_text"])


if __name__ == "__main__":
    main()
