# ==========================================
# IMPORTS
# ==========================================

import os
import tempfile
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ==========================================
# PAGE CONFIG
# ==========================================

load_dotenv()

st.set_page_config(
    page_title="ACE RAG Assistant",
    page_icon="A",
    layout="wide"
)

# ==========================================
# UI STYLING
# ==========================================

st.markdown("""
<style>

.stApp {
    background-image: url("https://4kwallpapers.com/images/walls/thumbs_2t/24785.png");
    background-size: cover;
    background-attachment: fixed;
}

.main {
    background: rgba(0,0,0,0.65);
    padding:20px;
    border-radius:15px;
}

h1,h2,h3,h4,h5,h6,p,label {
    color:white !important;
}

div.stButton > button {
    background: linear-gradient(135deg,#667eea,#764ba2);
    color:white;
    border:none;
    border-radius:10px;
    padding:10px 20px;
    font-weight:bold;
    transition:0.3s;
}

div.stButton > button:hover {
    transform:scale(1.05);
    box-shadow:0 0 10px rgba(118,75,162,0.7);
}

[data-testid="stChatMessage"] {
    border-radius:15px;
    background:rgba(255,255,255,0.08);
    padding:10px;
}

section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0f2027,#203a43,#2c5364);
}

section[data-testid="stSidebar"] * {
    color:white !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<h1 style='text-align:center;font-size:45px'>
🤖 Ace RAG AI Assistant
</h1>

<p style='text-align:center;font-size:20px'>
Ask questions from your documents instantly
</p>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Settings")

    api_key_input = st.text_input("Groq API Key", type="password")

    top_k = st.slider(
        "Chunks Retrieved",
        1,
        10,
        5
    )

    session_id = st.text_input("Session ID", value="default")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = {}
        st.rerun()

    st.markdown("---")
    st.caption("Upload PDFs → Ask Questions → AI Answers")


api_key = api_key_input or os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("Enter your Groq API Key")
    st.stop()


# ==========================================
# EMBEDDINGS + LLM
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)


# ==========================================
# FILE UPLOAD
# ==========================================

st.markdown("### 📂 Upload Documents")

uploaded_files = st.file_uploader(
    "",
    type="pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload PDFs to start chatting.")
    st.stop()


# ==========================================
# LOAD DOCUMENTS
# ==========================================

all_docs = []
tmp_paths = []

for pdf in uploaded_files:

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    tmp.write(pdf.getvalue())
    tmp.close()

    tmp_paths.append(tmp.name)

    loader = PyPDFLoader(tmp.name)

    docs = loader.load()

    for d in docs:
        d.metadata["source_file"] = pdf.name

    all_docs.extend(docs)

st.success(f"Loaded {len(all_docs)} pages from {len(uploaded_files)} PDFs")


# ==========================================
# CLEAN TEMP FILES
# ==========================================

for p in tmp_paths:
    try:
        os.unlink(p)
    except:
        pass


# ==========================================
# SPLITTING
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150
)

splits = splitter.split_documents(all_docs)


# ==========================================
# VECTOR STORE
# ==========================================

INDEX_DIR = "chroma_index"

if os.path.exists(INDEX_DIR):

    vectorstore = Chroma(
        persist_directory=INDEX_DIR,
        embedding_function=embeddings
    )

else:

    vectorstore = Chroma.from_documents(
        splits,
        embeddings,
        persist_directory=INDEX_DIR
    )


retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": top_k, "fetch_k": 20}
)


# ==========================================
# PROMPTS
# ==========================================

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system","Rewrite the user's question as a standalone query."),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
])


qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
    "Answer ONLY from the provided context.\n"
    "If answer not present reply:\n"
    "'Out of scope - not found in documents.'\n\n"
    "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
])


# ==========================================
# CHAT HISTORY
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}


def get_history(session):

    if session not in st.session_state.chat_history:
        st.session_state.chat_history[session] = ChatMessageHistory()

    return st.session_state.chat_history[session]


# ==========================================
# CHAT INPUT
# ==========================================

user_question = st.chat_input("Ask a question about your PDFs...")


def join_docs(docs, max_chars=7000):

    text = ""
    total = 0

    for d in docs:

        if total + len(d.page_content) > max_chars:
            break

        text += d.page_content + "\n\n---\n\n"
        total += len(d.page_content)

    return text


# ==========================================
# CHAT PIPELINE
# ==========================================

if user_question:

    history = get_history(session_id)

    st.chat_message("user").write(user_question)

    start = time.time()

    rewrite_msgs = rewrite_prompt.format_messages(
        chat_history=history.messages,
        input=user_question
    )

    standalone_query = llm.invoke(rewrite_msgs).content.strip()

    docs = retriever.invoke(standalone_query)

    context = join_docs(docs)

    qa_msgs = qa_prompt.format_messages(
        chat_history=history.messages,
        input=user_question,
        context=context
    )

    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        for chunk in llm.stream(qa_msgs):

            full_response += chunk.content
            placeholder.markdown(full_response)

    answer = full_response

    history.add_user_message(user_question)
    history.add_ai_message(answer)

    end = time.time()


    # ======================================
    # METRICS
    # ======================================

    col1,col2,col3 = st.columns(3)

    col1.metric("Chunks Indexed", len(splits))
    col2.metric("Retrieved Docs", len(docs))
    col3.metric("Response Time", f"{round(end-start,2)}s")


    # ======================================
    # SOURCES
    # ======================================

    sources=set()

    for d in docs:

        src=f"{d.metadata.get('source_file')} (Page {d.metadata.get('page','?')})"
        sources.add(src)

    if sources:

        st.markdown("### 📚 Sources")

        cols = st.columns(3)

        for i,s in enumerate(sources):

            cols[i%3].info(s)


    # ======================================
    # DEBUG
    # ======================================

    with st.expander("🧪 Debug Retrieval"):

        st.write("Standalone Query")
        st.code(standalone_query)

        st.write(f"Retrieved {len(docs)} chunks")


    with st.expander("📑 Retrieved Chunks"):

        for i,d in enumerate(docs,1):

            st.markdown(
                f"**{i}. {d.metadata.get('source_file')} (Page {d.metadata.get('page','?')})**"
            )

            st.write(d.page_content[:500])


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<hr>
<center>
            Developed By Mustafa
</center>
""", unsafe_allow_html=True)