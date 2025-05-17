# 🧭 Codebase Navigator using RAG

A Streamlit-powered application that allows you to analyze any GitHub repository using Retrieval-Augmented Generation (RAG). Simply enter the repository URL, and ask questions about the codebase — powered by `llama-index` and `ChromaDB`.

---

## 🚀 Features

- 🔍 Clone and index any public GitHub repository
- 🧠 Ask natural language questions about the codebase
- 🗂️ Load from existing database or clear and re-index
- 💬 AI-generated answers based on code context
- 🦙 Uses `llama-index` for document indexing and querying
- 🟣 Vector storage using `ChromaDB`

---

## 📦 Requirements

- Python 3.10+
- GitHub Access Token
- Google API Key (optional, for language detection)

Install dependencies:

```bash
pip install -r requirements.txt
```

🛠️ Setup

Clone this repo:

    git clone https://github.com/YOUR_USERNAME/Codebase-Navigator-using-RAG.git
    cd Codebase-Navigator-using-RAG

Create and activate virtual environment:

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

Add your GitHub Token and Google API Key in secrets.toml or via the Streamlit sidebar.

Run the app:

    streamlit run streamlit_app.py

📁 Project Structure

    .
    ├── chroma_db/               # ChromaDB storage
    ├── streamlit_app.py         # Main Streamlit app
    ├── final_rag.py             # RAG-related logic
    ├── rag.py                   # Helper functions for repo processing
    ├── server.py                # Optional server script
    ├── requirements.txt         # Python dependencies
    ├── saved_repos.pkl          # Saved metadata
    └── .gitignore

✨ Example Usage

Paste a GitHub URL like:

    https://github.com/YonkoSam/whatsapp-python-chatbot

Choose the branch (e.g., main)

Select an action:

- Add New Repository
- Load Existing Database
- Clear and Add Repository

Ask questions like:

    How is message routing handled?

    Where is the main entry point of the app?

🔐 Security Note

Don't commit your API keys or tokens. Use .streamlit/secrets.toml or environment variables to store credentials securely.
