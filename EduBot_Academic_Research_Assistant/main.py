import os
import streamlit as st
import pickle
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Load API key from Streamlit secrets or .env
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()

st.set_page_config(page_title="EduBot: Academic Research Assistant 📚", page_icon="📚", layout="wide")

st.markdown("""
    <style>
    .stApp {background: linear-gradient(to right, #141e30, #243b55); color: #ffffff;}
    .title {font-size: 2.8em; font-weight: bold; text-align: center; color: #00c6ff;}
    .subtitle {font-size: 1.2em; text-align: center; color: #eeeeee;}
    .stButton > button {background-color: #00c6ff; color: #000; border-radius: 8px; padding: 0.6em 1.4em; font-weight: bold;}
    .stButton > button:hover {background-color: #7df9ff;}
    .footer {text-align: center; color: #bbbbbb; margin-top: 30px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">EduBot: Academic Research Assistant 📚</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Summarize & Query Research Papers Instantly</div>', unsafe_allow_html=True)

# Sidebar for file upload
with st.sidebar:
    st.subheader("📂 Upload PDF/Text Files")
    uploaded_files = st.file_uploader("Upload Files", type=["pdf", "txt"], accept_multiple_files=True)
    process_file_clicked = st.button("🚀 Process Uploaded Files")

file_path = "edu_faiss_store.pkl"
progress_placeholder = st.empty()
llm = ChatOpenAI(temperature=0.2, max_tokens=1500, model_name="gpt-3.5-turbo")

# Process Uploaded Files
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
                docs = [Document(page_content=text, metadata={"source": file_name})]
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
