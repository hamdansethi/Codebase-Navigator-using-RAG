import streamlit as st
import os
import re
import shutil
import uuid
import logging
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.settings import Settings
from llama_index.core.node_parser import CodeSplitter, SimpleNodeParser
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
import chromadb
from github import Github
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Streamlit app
st.title("Dynamic GitHub Repository Q&A App")
st.write("Add GitHub repositories to an existing database, load previous data, or start fresh, and ask questions about the indexed codebases.")

# Sidebar for configuration
st.sidebar.header("Configuration")
github_token = st.sidebar.text_input("GitHub Token", type="password", value=os.environ.get("GITHUB_TOKEN", ""))
google_api_key = st.sidebar.text_input("Google API Key", type="password", value=os.environ.get("GOOGLE_API_KEY", ""))
google_model = st.sidebar.text_input("Google Model", value=os.environ.get("GOOGLE_MODEL", "gemini-pro"))
google_embedding_model = st.sidebar.text_input("Google Embedding Model", value=os.environ.get("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"))
persist_dir = st.sidebar.text_input("Chroma DB Directory", value="./chroma_db")
clear_db = st.sidebar.checkbox("Clear Existing Database", value=False)

# Initialize settings
if google_api_key:
    try:
        Settings.llm = GoogleGenAI(model=google_model, temperature=0.1, api_key=google_api_key)
        Settings.embed_model = GoogleGenAIEmbedding(model=google_embedding_model, api_key=google_api_key)
    except Exception as e:
        st.error(f"Error initializing Google API: {str(e)}")

# Language and extension mappings
languages = [
    "python", "javascript", "java", "c", "cpp", "php", "ruby", "rust", "go", "typescript",
    "swift", "kotlin", "scala", "bash", "sql", "html", "css", "lua", "elisp", "markdown",
    "yaml", "json", "perl", "r", "objc", "haskell", "toml", "fortran", "hack", "commonlisp"
]
ext_to_lang = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".java": "java", ".c": "c",
    ".cpp": "cpp", ".php": "php", ".rb": "ruby", ".rs": "rust", ".go": "go", ".swift": "swift",
    ".kt": "kotlin", ".scala": "scala", ".sh": "bash", ".sql": "sql", ".html": "html",
    ".css": "css", ".lua": "lua", ".el": "elisp", ".md": "markdown", ".yaml": "yaml",
    ".yml": "yaml", ".json": "json", ".pl": "perl", ".r": "r", ".m": "objc",
    ".hs": "haskell", ".toml": "toml", ".f90": "fortran", ".f": "fortran", ".hack": "hack",
    ".lisp": "commonlisp"
}

# Create splitters
splitters = {lang: CodeSplitter(language=lang, chunk_lines=40, chunk_lines_overlap=15, max_chars=1500) for lang in languages}
fallback_splitter = SimpleNodeParser()

# Function to validate repository URL
def validate_repo_url(repo_url):
    pattern = r'^https://github\.com/[\w-]+/[\w-]+$'
    return bool(re.match(pattern, repo_url))

# Function to process repository and return nodes
def process_repository(repo_url, github_token):
    if not validate_repo_url(repo_url):
        st.error("Invalid GitHub repository URL. Use format: https://github.com/owner/repository")
        return []
    
    try:
        st.write(f"Processing {repo_url}...")
        parts = repo_url.replace("https://github.com/", "").split("/")
        owner, repository = parts[0], parts[1]
        
        github_client = GithubClient(github_token=github_token, verbose=True)
        pygithub = Github(github_token)
        repo = pygithub.get_repo(f"{owner}/{repository}")
        latest_commit_sha = repo.get_branch("main").commit.sha
        st.write(f"Fetched repo: {owner}/{repository}")

        reader = GithubRepositoryReader(github_client=github_client, owner=owner, repo=repository)
        documents = reader.load_data(branch="main")
        st.write(f"Loaded {len(documents)} documents")

        all_nodes = []
        for doc in documents:
            filename = doc.metadata.get("filename", "")
            _, ext = os.path.splitext(filename.lower())
            lang = ext_to_lang.get(ext)
            splitter = splitters.get(lang, fallback_splitter)
            nodes = splitter.get_nodes_from_documents([doc])
            for node in nodes:
                node.metadata["repo_url"] = repo_url
                node.metadata["node_id"] = str(uuid.uuid4())
            all_nodes.extend(nodes)
        
        st.write(f"Processed {len(all_nodes)} nodes")
        return all_nodes
    except Exception as e:
        st.error(f"Error processing repository {repo_url}: {str(e)}")
        logger.error(f"Repository processing error: {str(e)}")
        return []

# Function to initialize or load Chroma database
def initialize_chroma(persist_dir, clear_db=False):
    try:
        # Clear existing database if requested
        if clear_db and os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            st.write(f"Cleared existing database at {persist_dir}")

        # Ensure directory exists and is writable
        os.makedirs(persist_dir, exist_ok=True)
        if not os.access(persist_dir, os.W_OK):
            raise PermissionError(f"Directory {persist_dir} is not writable")
        
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        collection = chroma_client.get_or_create_collection(name="MyExternalContext")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return chroma_client, collection, vector_store, storage_context
    except Exception as e:
        st.error(f"Error initializing Chroma database: {str(e)}")
        logger.error(f"Chroma initialization error: {str(e)}")
        return None, None, None, None

# Function to load or create index
def load_or_create_index(persist_dir, new_nodes=None, clear_db=False):
    try:
        chroma_client, collection, vector_store, storage_context = initialize_chroma(persist_dir, clear_db)
        if not chroma_client or not collection:
            return None, None

        if new_nodes:
            st.write(f"Indexing {len(new_nodes)} nodes...")
            index = VectorStoreIndex(
                nodes=new_nodes,
                storage_context=storage_context,
                insert_batch_size=2048
            )
        else:
            st.write("Loading existing index...")
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
        return index, collection
    except Exception as e:
        st.error(f"Error loading/creating index: {str(e)}")
        logger.error(f"Index creation/loading error: {str(e)}")
        return None, None

# Main app logic
st.header("Repository Management")
repo_url = st.text_input("GitHub Repository URL", value="https://github.com/YonkoSam/whatsapp-python-chatbot")
action = st.radio("Action", ["Add New Repository", "Load Existing Database", "Clear and Add Repository"])

# Initialize index and collection in session state
if 'index' not in st.session_state:
    st.session_state.index = None
if 'collection' not in st.session_state:
    st.session_state.collection = None

if st.button("Execute"):
    if not github_token or not google_api_key:
        st.error("Please provide GitHub Token and Google API Key in the sidebar.")
    else:
        with st.spinner("Processing..."):
            if action == "Add New Repository":
                new_nodes = process_repository(repo_url, github_token)
                if new_nodes:
                    st.session_state.index, st.session_state.collection = load_or_create_index(persist_dir, new_nodes)
                    if st.session_state.index:
                        st.success(f"Repository {repo_url} added! Total documents in collection: {st.session_state.collection.count()}")
                    else:
                        st.error("Failed to create index.")
                else:
                    st.error("No nodes processed from repository.")
            elif action == "Load Existing Database":
                st.session_state.index, st.session_state.collection = load_or_create_index(persist_dir)
                if st.session_state.index:
                    st.success(f"Database loaded! Total documents in collection: {st.session_state.collection.count()}")
                else:
                    st.error("Failed to load database.")
            else:  # Clear and Add Repository
                new_nodes = process_repository(repo_url, github_token)
                if new_nodes:
                    st.session_state.index, st.session_state.collection = load_or_create_index(persist_dir, new_nodes, clear_db=True)
                    if st.session_state.index:
                        st.success(f"Database cleared and repository {repo_url} added! Total documents in collection: {st.session_state.collection.count()}")
                    else:
                        st.error("Failed to create index.")
                else:
                    st.error("No nodes processed from repository.")

# Query section
st.header("Ask Questions")
query = st.text_input("Enter your question", value="How does the WhatsApp bot handle message routing?")
if st.button("Ask"):
    if st.session_state.index is None:
        st.error("Please process a repository or load a database first.")
    else:
        with st.spinner("Generating response..."):
            try:
                metadata_postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
                reranker = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base")
                query_engine = st.session_state.index.as_query_engine(
                    similarity_top_k=5,
                    node_postprocessors=[metadata_postproc, reranker]
                )
                response = query_engine.query(query)
                st.write("**Response:**")
                st.write(str(response))
            except Exception as e:
                st.error(f"Error querying: {str(e)}")
                logger.error(f"Query error: {str(e)}")