# 🤖 Abeer AI — Intelligent AI Document Assistant

Abeer AI is a local AI-powered document assistant that allows users to upload PDF documents and ask questions about their content using a Retrieval-Augmented Generation (RAG) pipeline.

The application uses **Llama 3.2 through Ollama** for local AI responses, **FAISS** for vector similarity search, and **Nomic Embed Text** for document embeddings.

## 🚀 Features

* 📄 PDF document upload
* 💬 Natural-language document Q&A
* 🧠 Retrieval-Augmented Generation (RAG)
* 🔎 Semantic document search
* 🗂️ FAISS vector database
* 🤖 Local Llama 3.2 LLM
* 🔤 Nomic Embed Text embeddings
* 📚 Source page references
* 💭 Conversation memory
* 📊 Document statistics
* 💡 Suggested questions
* 🎨 Dark/Light theme
* 🎨 Custom accent colors
* 🗑️ Clear conversation
* 🔒 Local AI processing with Ollama

## 🛠️ Technologies Used

| Technology       | Purpose                          |
| ---------------- | -------------------------------- |
| Python           | Application development          |
| Streamlit        | Web interface                    |
| LangChain        | RAG pipeline and LLM integration |
| Ollama           | Local AI model runtime           |
| Llama 3.2        | Local language model             |
| Nomic Embed Text | Document embeddings              |
| FAISS            | Vector similarity search         |
| PyPDF            | PDF document processing          |

## 🧠 How It Works

```text
                    ┌─────────────────┐
                    │   User Uploads  │
                    │      PDF        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  PDF Extraction │
                    │    PyPDFLoader  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Text Chunking   │
                    │ LangChain       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Embeddings    │
                    │ Nomic Embed Text│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ FAISS Vector DB │
                    └────────┬────────┘
                             │
             User Question  │
                    ─────────┤
                             ▼
                    ┌─────────────────┐
                    │ Semantic Search │
                    │ Top-K Chunks    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Llama 3.2     │
                    │  via Ollama     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  AI Response +  │
                    │ Source Pages    │
                    └─────────────────┘
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Abeer-AI
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

## 🤖 Install Ollama

Abeer AI uses Ollama to run the AI models locally.

Install Ollama and then download the required models:

```powershell
ollama pull llama3.2
```

```powershell
ollama pull nomic-embed-text
```

Make sure Ollama is running before starting the application.

## ▶️ Run the Application

Start the Streamlit application:

```powershell
streamlit run app.py
```

The application will open in your browser.

## 📄 Using Abeer AI

1. Start the application.
2. Upload a PDF from the sidebar.
3. Wait for the document to be processed.
4. Ask a question about the document.
5. Abeer AI retrieves relevant document content.
6. Llama 3.2 generates the answer using the retrieved context.
7. Source page references are displayed with the response.

## 🔐 Privacy

Abeer AI is designed around local AI processing.

Documents are processed locally through the application, while the LLM and embedding models run through Ollama on the user's machine.

Private documents should not be committed to the GitHub repository.

## 📁 Project Structure

```text
Abeer-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│   └── logo.png
│
└── data/
    └── .gitkeep
```

## 🔮 Future Improvements

Potential future enhancements include:

* Multiple document support
* Persistent vector databases
* Improved retrieval and reranking
* Additional local LLM models
* Document type expansion
* Authentication
* Conversation export
* Advanced source citations
* Deployment support

## 👨‍💻 Author

**Muhammad Abdullah**

Data Science Graduate | AI & Machine Learning

Lahore, Pakistan

* LinkedIn: linkedin.com/in/muhammad-abdullah-394288242
* GitHub: github.com/MuhammadAbdullahIkram

## ⭐ Project

If you find this project useful, consider giving the repository a star.
