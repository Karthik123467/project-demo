import os
import streamlit as st
import pickle

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, TextLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load API key from secrets or .env
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()

st.set_page_config(page_title="EduBot: Academic Paper Research", page_icon="📚", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stApp {background: linear-gradient(to right, #141e30, #243b55); color: #ffffff;}
    .title {font-size: 2.8em; font-weight: bold; text-align: center; margin-bottom: 5px; color: #00c6ff;}
    .subtitle {font-size: 1.1em; text-align: center; color: #eeeeee; margin-bottom: 30px;}
    .stButton > button {background-color: #00c6ff; color: #000000; border-radius: 8px; padding: 0.6em 1.4em; font-weight: bold; border: none;}
    .stButton > button:hover {background-color: #7df9ff;}
    .footer {position: relative; text-align: center; font-size: 0.9em; margin-top: 50px; color: #bbbbbb;}
    </style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown('<div class="title">EduBot: Academic Research Assistant 📚</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Summarize & Query Research Papers Instantly</div>', unsafe_allow_html=True)

# Sidebar for URLs and File Upload
with st.sidebar:
    st.subheader("📑 Enter Research Paper URLs")
    urls = [st.text_input(f"🔗 Paper URL {i+1}") for i in range(3)]
    process_url_clicked = st.button("🚀 Process Papers from URLs")
    st.markdown("---")
    st.subheader("📂 Or Upload PDF/Text Files")
    uploaded_files = st.file_uploader("Upload Files", type=["pdf", "txt"], accept_multiple_files=True)
    process_file_clicked = st.button("🚀 Process Uploaded Files")

file_path = "edu_faiss_store.pkl"
progress_placeholder = st.empty()
llm = ChatOpenAI(temperature=0.2, max_tokens=1500, model_name="gpt-3.5-turbo")


# URL Processing
if process_url_clicked:
    with st.spinner("Fetching and processing papers from URLs..."):
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        if not data or all(not d.page_content.strip() for d in data):
            st.error("❌ No valid text found from the URLs. Check links or try file upload.")
            st.stop()

        progress_placeholder.progress(33, "Loaded Papers ✅")

        text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=3000)
        docs = text_splitter.split_documents(data)
        if not docs:
            st.error("❌ No text chunks were created from the papers.")
            st.stop()

        progress_placeholder.progress(66, "Split Text ✅")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        progress_placeholder.progress(100, "Vectorstore Built ✅")

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        st.success("✅ Papers Processed Successfully from URLs!")

# File Upload Processing
if process_file_clicked:
    if not uploaded_files:
        st.error("❌ Please upload at least one file.")
        st.stop()

    all_docs = []
    with st.spinner("Processing uploaded files..."):
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            if file_name.endswith(".txt"):
                text = uploaded_file.read().decode("utf-8")
                loader = TextLoader(file_name)
                docs = [loader.load(text)]
            elif file_name.endswith(".pdf"):
                with open(file_name, "wb") as f:
                    f.write(uploaded_file.read())
                loader = PyPDFLoader(file_name)
                docs = loader.load()
                os.remove(file_name)
            else:
                continue
            all_docs.extend(docs)

        if not all_docs:
            st.error("❌ Failed to process any uploaded files.")
            st.stop()

        progress_placeholder.progress(33, "Loaded Files ✅")
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=3000)
        docs = text_splitter.split_documents(all_docs)
        if not docs:
            st.error("❌ No text chunks were created from the uploaded files.")
            st.stop()

        progress_placeholder.progress(66, "Split Text ✅")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        progress_placeholder.progress(100, "Vectorstore Built ✅")

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        st.success("✅ Files Processed Successfully!")

# Query Section
query = st.text_input("💬 Ask Your Academic Question:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), return_source_documents=True
        )

        with st.spinner("Generating Answer..."):
            result = chain.invoke({"query": query})

        st.header("🔎 Answer")
        st.success(result["result"])

        source_docs = result.get("source_documents", [])
        unique_sources = set()
        for doc in source_docs:
            source = doc.metadata.get("source", "")
            if source:
                unique_sources.add(source)

        if unique_sources:
            with st.expander("📂 Sources Used"):
                for src in unique_sources:
                    st.write(src)
        else:
            st.info("No sources found for this answer.")

# Footer
st.markdown('<div class="footer">🚀 Created by PKK</div>', unsafe_allow_html=True)
