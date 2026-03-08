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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


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

    top_k = st.slider("Chunks Retrieved", 1, 10, 5)

    session_id = st.text_input("Session ID", value="default")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = {}
        st.session_state.messages = []
        st.rerun()

    if st.button("♻️ Reset Vector Store"):
        if "vectorstore" in st.session_state:
            del st.session_state["vectorstore"]
            del st.session_state["splits"]
        st.rerun()

    st.markdown("---")
    st.caption("Upload PDFs → Ask Questions → AI Answers")


api_key = api_key_input or os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("Enter your Groq API Key")
    st.stop()


# ==========================================
# SESSION STATE INIT
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "splits" not in st.session_state:
    st.session_state.splits = []


# ==========================================
# EMBEDDINGS + LLM (cached)
# ==========================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

embeddings = load_embeddings()


@st.cache_resource
def load_llm(_api_key):
    return ChatGroq(groq_api_key=_api_key, model_name="llama-3.3-70b-versatile")

llm = load_llm(api_key)


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
# LOAD + INDEX DOCUMENTS
# Only re-index when files actually change
# ==========================================

file_key = "_".join(sorted([f.name + str(f.size) for f in uploaded_files]))

if st.session_state.get("file_key") != file_key or st.session_state.vectorstore is None:

    with st.spinner("📖 Reading and indexing your documents..."):

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

        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150
        )
        splits = splitter.split_documents(all_docs)

        vectorstore = FAISS.from_documents(splits, embeddings)

        st.session_state.vectorstore = vectorstore
        st.session_state.splits = splits
        st.session_state.file_key = file_key
        st.session_state.all_docs = all_docs

    st.success(f"✅ Indexed {len(splits)} chunks from {len(all_docs)} pages across {len(uploaded_files)} PDF(s)")

else:
    splits = st.session_state.splits
    vectorstore = st.session_state.vectorstore
    st.success(f"✅ Using cached index — {len(splits)} chunks ready")


# ==========================================
# RETRIEVER
# ==========================================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": top_k, "fetch_k": min(20, len(splits))}
)


# ==========================================
# PROMPTS
# ==========================================

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's question as a standalone query."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer ONLY from the provided context.\n"
     "If answer not present reply:\n"
     "'Out of scope - not found in documents.'\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])


# ==========================================
# HELPERS
# ==========================================

def get_history(session):
    if session not in st.session_state.chat_history:
        st.session_state.chat_history[session] = ChatMessageHistory()
    return st.session_state.chat_history[session]


def join_docs(docs, max_chars=7000):
    text, total = "", 0
    for d in docs:
        if total + len(d.page_content) > max_chars:
            break
        text += d.page_content + "\n\n---\n\n"
        total += len(d.page_content)
    return text


# ==========================================
# RENDER PAST MESSAGES
# ==========================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ==========================================
# CHAT INPUT + PIPELINE
# ==========================================

user_question = st.chat_input("Ask a question about your PDFs...")

if user_question:

    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    history = get_history(session_id)
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
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    history.add_user_message(user_question)
    history.add_ai_message(full_response)

    end = time.time()

    col1, col2, col3 = st.columns(3)
    col1.metric("Chunks Indexed", len(splits))
    col2.metric("Retrieved Docs", len(docs))
    col3.metric("Response Time", f"{round(end - start, 2)}s")

    sources = set()
    for d in docs:
        src = f"{d.metadata.get('source_file')} (Page {d.metadata.get('page', '?')})"
        sources.add(src)

    if sources:
        st.markdown("### 📚 Sources")
        cols = st.columns(3)
        for i, s in enumerate(sources):
            cols[i % 3].info(s)

    with st.expander("🧪 Debug Retrieval"):
        st.write("Standalone Query")
        st.code(standalone_query)
        st.write(f"Retrieved {len(docs)} chunks")

    with st.expander("📑 Retrieved Chunks"):
        for i, d in enumerate(docs, 1):
            st.markdown(f"**{i}. {d.metadata.get('source_file')} (Page {d.metadata.get('page', '?')})**")
            st.write(d.page_content[:500])


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<hr>
<center>Developed By Mustafa</center>
""", unsafe_allow_html=True)
