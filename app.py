import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
import tempfile, time, random
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

def safe_refine_chunk(args):
    llm, prompt_template, original_chunk, max_retries = args
    prompt = prompt_template.format(chunk=original_chunk.page_content)
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return Document(page_content=response.content.strip(), metadata=original_chunk.metadata)
        except Exception:
            time.sleep(2 ** attempt + random.uniform(0.5, 1.5))
    return original_chunk

def process_pdf(pdf_file):
    with st.spinner("Processing PDF…"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()

        total = sum(len(d.page_content) for d in docs)
        size = max(total // 10, 300)
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=300)
        raw_chunks = splitter.split_documents(docs)

        refiner = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        prompt_refine = PromptTemplate(
            input_variables=["chunk"],
            template='''You are a helpful assistant refining document chunks…
Original:
"""{chunk}"""'''
        )

        with ThreadPoolExecutor() as executor:
            refined_chunks = list(executor.map(safe_refine_chunk, [(refiner, prompt_refine, chunk, 5) for chunk in raw_chunks]))

        emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vs = FAISS.from_documents(refined_chunks, emb)
        retriever = vs.as_retriever(search_kwargs={"k": 4})

        qa_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=qa_llm,
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain

def main():
    st.set_page_config(page_title="PDF Chatbot", layout="wide")
    st.title("📄 Chat with your PDF")

    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("qa_chain", None)
    st.session_state.setdefault("processed", False)

    pdf_file = st.file_uploader("Upload a PDF", type="pdf")

    if pdf_file and not st.session_state.processed:
        st.session_state.qa_chain = process_pdf(pdf_file)
        st.session_state.processed = True
        st.success("✅ PDF processed! Ask away!")

    if st.session_state.qa_chain:
        q = st.chat_input("Ask a question about your PDF…")
        if q:
            with st.spinner("Thinking…"):
                result = st.session_state.qa_chain.invoke({"question": q, "chat_history": st.session_state.chat_history})
                a = result["answer"]
            st.session_state.chat_history.append((q, a))

        for user_q, bot_a in st.session_state.chat_history:
            st.chat_message("user").write(user_q)
            st.chat_message("assistant").markdown(bot_a, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
