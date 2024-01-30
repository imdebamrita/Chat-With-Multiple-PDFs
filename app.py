import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
# from langchain.exceptions import BlockedPromptException
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        text = None
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_chunks_encoded = [chunk.encode(
        'utf-8', 'replace').decode('utf-8') for chunk in text_chunks]
    vector_store = FAISS.from_texts(text_chunks_encoded, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    return a very well formated markdown text with headings, sub headings, points etc.

    Answer the quistion as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n{context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", 'question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    # model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # safety_rating = model.rate_safety(user_question)
    # for rating in safety_rating:
    #     if rating.probability == "HIGH":
    #         st.warning(
    #             "Warning: This question may contain explicit or harmful content.")
    #         return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    try:
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True)

        print("response:", response)
        st.subheader("Answer:", divider=True)
        st.markdown(response["output_text"])
    except Exception as e:
        st.warning(
            f"Warning: This question may contain explicit or harmful content: \n{e}")


def main():
    st.set_page_config(page_title="Chat With Multiple PDF",
                       initial_sidebar_state='expanded',
                       menu_items={
                           'Get Help': 'https://www.linkedin.com/in/imdebamritapaul/',
                           'Report a bug': "mailto:imdebamrita@1gmail.com",
                           'About': "< ___Made by : Debamrita Paul___ > Connect in LinkedIn: https://www.linkedin.com/in/imdebamritapaul/ 📲🚀"
                       })
    st.header("📚 Chat with Multiple PDFs using Gemini Pro 🚀")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type=['pdf'])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # if pdf_docs is not None:
                #     print(type(pdf_docs))
                #     st.write("Uploaded Files:", pdf_docs)

                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done and Go Ahead to ask any Question")
                # else:
                #     st.warning("No files uploaded. Please upload PDF files.")
        st.sidebar.text("Made by Debamrita Paul")
        st.sidebar.write(
            "[Connect ⤴](https://www.linkedin.com/in/imdebamritapaul/)")


if __name__ == "__main__":
    main()
