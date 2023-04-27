from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import streamlit as st
import os
def main():
    st.title("PDF GPT creator")
    menu = ['Home','About']
    choice = st.sidebar.selectbox('Menu',menu)

    if choice == "Home":
        st.subheader('Home')
        pdf_file = st.file_uploader('Upload a PDF file',type=['pdf','docx','txt'])
        if pdf_file:
            train_model(pdf_file)
    if choice == 'About':
        st.write("This is an app that takes a pdf file as input data and we can use that data over a GPT and get information as answers by questioning.")
def train_model(pdf_file):
    reader = PdfReader(pdf_file)
    raw_text = ''
    for i,page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    text_splitter = CharacterTextSplitter(separator = '\n',chunk_size=1000,chunk_overlap=200,length_function=len)
    texts = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type='stuff')
    query = st.text_input('question here...')
    docs = docsearch.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    st.write(response)
if __name__ == '__main__':
    main()
