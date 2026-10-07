import streamlit as st
import tempfile
import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Abeer AI",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

if "accent_color" not in st.session_state:
    st.session_state.accent_color = "#6C63FF"


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():

    return ChatOllama(
        model="llama3.2",
        temperature=0.7
    )


@st.cache_resource
def load_embeddings():

    return OllamaEmbeddings(
        model="nomic-embed-text"
    )


# =========================================================
# LOAD AI MODELS SAFELY
# =========================================================

try:

    model = load_model()
    embeddings = load_embeddings()

    models_ready = True

except Exception as e:

    models_ready = False
    model = None
    embeddings = None

    st.error(
        "⚠️ Abeer AI could not connect to Ollama. "
        "Please make sure Ollama is running."
    )


# =========================================================
# CREATE VECTOR DATABASE
# =========================================================

def create_vectorstore(uploaded_file):

    if embeddings is None:
        raise RuntimeError(
            "Embedding model is not available."
        )

    pdf_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(
                uploaded_file.getbuffer()
            )

            pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(
            pdf_path
        )

        documents = loader.load()

        if not documents:

            raise ValueError(
                "The uploaded PDF does not contain readable content."
            )

        # Split document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = text_splitter.split_documents(
            documents
        )

        if not chunks:

            raise ValueError(
                "No readable text was found in the PDF."
            )

        # Create FAISS vector database
        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        # Save statistics
        st.session_state.page_count = len(
            documents
        )

        st.session_state.chunk_count = len(
            chunks
        )

        return vectorstore

    finally:

        if (
            pdf_path is not None
            and os.path.exists(pdf_path)
        ):

            os.unlink(pdf_path)


# =========================================================
# RAG PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_template(
    """
You are Abeer AI, an intelligent AI document assistant.

Answer the user's question using the provided document context
and relevant conversation history.

Rules:
- Use the document context as the primary source of truth.
- Use conversation history only to understand follow-up questions.
- Do not invent information.
- Give clear and concise answers.
- If the required information is not available in the document context, say:

"I couldn't find this information in the provided document."

Conversation History:
{history}

Document Context:
{context}

Current Question:
{question}

Answer:
"""
)


# =========================================================
# CUSTOM THEME
# =========================================================

accent = st.session_state.accent_color

if st.session_state.theme == "Dark":

    background = "#0E1117"
    card_background = "#161B22"
    text_color = "#FFFFFF"
    secondary_text = "#B8C1CC"
    sidebar_background = "#11151C"

else:

    background = "#FFFFFF"
    card_background = "#F5F7FA"
    text_color = "#111827"
    secondary_text = "#4B5563"
    sidebar_background = "#F8FAFC"


# =========================================================
# CSS
# =========================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {background};
        color: {text_color};
    }}

    [data-testid="stSidebar"] {{
        background-color: {sidebar_background};
    }}

    h1, h2, h3, h4, p, label {{
        color: {text_color};
    }}

    .subtitle {{
        color: {secondary_text};
        font-size: 17px;
        margin-bottom: 20px;
    }}

    .card {{
        background-color: {card_background};
        padding: 20px;
        border-radius: 14px;
        border: 1px solid {accent};
        margin-bottom: 15px;
    }}

    .source-box {{
        background-color: {card_background};
        border-left: 4px solid {accent};
        padding: 10px;
        border-radius: 8px;
        margin-top: 10px;
    }}

    .accent-line {{
        height: 4px;
        background-color: {accent};
        border-radius: 5px;
        margin-bottom: 25px;
    }}

    div.stButton > button {{
        border-radius: 10px;
        border: 1px solid {accent};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚙️ Abeer AI")

    st.markdown("---")

    # -----------------------------------------------------
    # APPEARANCE
    # -----------------------------------------------------

    st.subheader("🎨 Appearance")

    theme = st.radio(
        "Theme",
        ["Dark", "Light"],
        index=(
            0
            if st.session_state.theme == "Dark"
            else 1
        )
    )

    st.session_state.theme = theme

    color_options = {
        "Purple": "#6C63FF",
        "Blue": "#2196F3",
        "Green": "#22C55E",
        "Orange": "#F97316",
        "Pink": "#EC4899",
        "Cyan": "#06B6D4"
    }

    selected_color = st.selectbox(
        "Accent Color",
        list(color_options.keys())
    )

    use_custom = st.checkbox(
        "Use Custom Color"
    )

    if use_custom:

        custom_color = st.color_picker(
            "Choose Color",
            st.session_state.accent_color
        )

        st.session_state.accent_color = (
            custom_color
        )

    else:

        st.session_state.accent_color = (
            color_options[selected_color]
        )

    st.markdown("---")

    # -----------------------------------------------------
    # PDF UPLOAD
    # -----------------------------------------------------

    st.subheader("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if (
            st.session_state.uploaded_filename
            != uploaded_file.name
        ):

            try:

                with st.spinner(
                    "🔄 Processing document..."
                ):

                    st.session_state.vectorstore = (
                        create_vectorstore(
                            uploaded_file
                        )
                    )

                    st.session_state.uploaded_filename = (
                        uploaded_file.name
                    )

                    st.session_state.messages = []

                st.success(
                    "✅ Document processed successfully!"
                )

            except Exception as e:

                st.session_state.vectorstore = None

                st.session_state.uploaded_filename = None

                st.error(
                    "❌ Could not process this PDF."
                )

                st.caption(
                    f"Details: {str(e)}"
                )

    # -----------------------------------------------------
    # CURRENT DOCUMENT
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader("📑 Current Document")

    if st.session_state.uploaded_filename:

        st.write(
            f"**{st.session_state.uploaded_filename}**"
        )

    else:

        st.info(
            "No document uploaded."
        )

    # -----------------------------------------------------
    # DOCUMENT STATISTICS
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader("📊 Document Statistics")

    if st.session_state.vectorstore is not None:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "📄 Pages",
                st.session_state.page_count
            )

        with col2:

            st.metric(
                "🧩 Chunks",
                st.session_state.chunk_count
            )

        st.metric(
            "🔎 Retrieved Results",
            3
        )

    else:

        st.caption(
            "Upload a PDF to view document statistics."
        )

    # -----------------------------------------------------
    # MODEL INFORMATION
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader("🤖 AI Model")

    st.write("**Model:** Llama 3.2")
    st.write("**Embeddings:** nomic-embed-text")
    st.write("**Vector DB:** FAISS")
    st.write("**Framework:** LangChain")

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader("✨ Features")

    st.write("✅ Local LLM")
    st.write("✅ RAG")
    st.write("✅ FAISS")
    st.write("✅ PDF Q&A")
    st.write("✅ Conversation Memory")
    st.write("✅ Source References")

    # -----------------------------------------------------
    # CLEAR CHAT
    # -----------------------------------------------------

    st.markdown("---")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# MAIN HEADER
# =========================================================

st.title("🤖 Abeer AI")

st.markdown(
    '<div class="subtitle">'
    'Your Intelligent AI Document Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="accent-line"></div>',
    unsafe_allow_html=True
)

st.caption(
    "Powered by Llama 3.2 • RAG • FAISS • LangChain"
)


# =========================================================
# WELCOME SECTION
# =========================================================

if st.session_state.vectorstore is None:

    st.info(
        "👋 Welcome to Abeer AI! "
        "Upload a PDF from the sidebar to start asking questions."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="card">

            ### 📄 PDF Q&A

            Upload your PDF and ask questions
            directly from your document.

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">

            ### 🧠 Smart RAG

            Abeer AI retrieves relevant
            information before generating answers.

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="card">

            ### 🔒 Local AI

            Your document processing and
            AI model run locally with Ollama.

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# SUGGESTED QUESTIONS
# =========================================================

if st.session_state.vectorstore is not None:

    st.markdown("---")

    st.subheader("💡 Suggested Questions")

    q1, q2 = st.columns(2)

    with q1:

        if st.button(
            "📄 What is this document about?",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What is this document about?"
            )

    with q2:

        if st.button(
            "📝 Summarize the document",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "Summarize the document."
            )

    q3, q4 = st.columns(2)

    with q3:

        if st.button(
            "🔑 What are the key points?",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What are the key points of this document?"
            )

    with q4:

        if st.button(
            "📚 Explain the main topics",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "Explain the main topics covered in this document."
            )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            st.markdown(
                f"""
                <div class="source-box">
                📚 <b>Sources:</b> {message["sources"]}
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# CHAT INPUT
# =========================================================

chat_question = st.chat_input(
    "💬 Ask Abeer AI about your document..."
)


# =========================================================
# QUESTION SELECTION
# =========================================================

user_question = chat_question

if (
    user_question is None
    and st.session_state.pending_question is not None
):

    user_question = (
        st.session_state.pending_question
    )

    st.session_state.pending_question = None


# =========================================================
# PROCESS QUESTION
# =========================================================

if user_question:

    if st.session_state.vectorstore is None:

        st.warning(
            "⚠️ Please upload a PDF document first."
        )

    elif not models_ready:

        st.error(
            "❌ AI model is not available. "
            "Please make sure Ollama is running."
        )

    else:

        # -------------------------------------------------
        # USER MESSAGE
        # -------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):

            st.markdown(
                user_question
            )

        # -------------------------------------------------
        # ASSISTANT RESPONSE
        # -------------------------------------------------

        with st.chat_message("assistant"):

            try:

                # =========================================
                # RETRIEVE DOCUMENT CONTEXT
                # =========================================

                with st.spinner(
                    "🔎 Searching document..."
                ):

                    retriever = (
                        st.session_state.vectorstore
                        .as_retriever(
                            search_kwargs={
                                "k": 3
                            }
                        )
                    )

                    results = retriever.invoke(
                        user_question
                    )

                if not results:

                    raise ValueError(
                        "No relevant information was found."
                    )

                # =========================================
                # BUILD CONTEXT
                # =========================================

                context = "\n\n".join(
                    doc.page_content
                    for doc in results
                )

                # =========================================
                # BUILD CONVERSATION HISTORY
                # =========================================

                history = ""

                for message in (
                    st.session_state.messages[-6:]
                ):

                    role = (
                        message["role"]
                        .capitalize()
                    )

                    content = (
                        message["content"]
                    )

                    history += (
                        f"{role}: {content}\n"
                    )

                # =========================================
                # GENERATE ANSWER
                # =========================================

                with st.spinner(
                    "🤖 Abeer AI is thinking..."
                ):

                    final_prompt = prompt.invoke(
                        {
                            "history": history,
                            "context": context,
                            "question": user_question
                        }
                    )

                    response = model.invoke(
                        final_prompt
                    )

                    answer = response.content

                # =========================================
                # DISPLAY ANSWER
                # =========================================

                st.markdown(
                    answer
                )

                # =========================================
                # SOURCE REFERENCES
                # =========================================

                source_data = []

                for doc in results:

                    page_number = (
                        doc.metadata.get(
                            "page",
                            0
                        ) + 1
                    )

                    source_data.append(
                        page_number
                    )

                source_pages = sorted(
                    set(source_data)
                )

                sources_text = ", ".join(
                    f"Page {page}"
                    for page in source_pages
                )

                st.markdown(
                    f"""
                    <div class="source-box">
                    📚 <b>Sources:</b> {sources_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # =========================================
                # SAVE ASSISTANT MESSAGE
                # =========================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_text
                    }
                )

            except Exception as e:

                error_message = (
                    "❌ Sorry, I couldn't process your "
                    "question right now."
                )

                st.error(
                    error_message
                )

                st.caption(
                    f"Details: {str(e)}"
                )