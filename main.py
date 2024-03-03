from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

load_dotenv()

def main():
    st.set_page_config(page_title="Financial Analyzer")
    st.header("Ask your PDF")

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # Take the PDF file and read it
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        #st.write(text)
        # Split pdf into chunks
        textSplitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = textSplitter.split_text(text)
        #st.write(chunks)
        
        # Create Embeddings
        embeddings = OpenAIEmbeddings()
        knowledgeBase = FAISS.from_texts(chunks, embeddings)

        # Show user input after uploading the file
        userQuestion = st.text_input("Ask your PDF")
        if userQuestion:
            docs = knowledgeBase.similarity_search(userQuestion)
            
            llm = OpenAI(temperature=0.9)
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=userQuestion)
                print(cb)
            st.write("AI Response:", response)
                                

if __name__ == '__main__':
    main()

#https://www.youtube.com/watch?v=wUAUdEw5oxM