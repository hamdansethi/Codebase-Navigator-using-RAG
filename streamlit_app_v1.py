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

# Function to validate and get valid branch
def get_valid_branch(repo, branch):
    try:
        repo.get_branch(branch)
        return branch
    except Exception:
        for fallback_branch in ["main", "master"]:
            try:
                repo.get_branch(fallback_branch)
                st.warning(f"Branch '{branch}' not found. Using '{fallback_branch}' instead.")
                return fallback_branch
            except Exception:
                continue
        return None

# Function to get existing repositories from Chroma
def get_existing_repos(collection):
    try:
        if collection is None:
            return []
        results = collection.get(include=["metadatas"])
        repo_urls = {item.get("repo_url") for item in results["metadatas"] if item.get("repo_url")}
        return sorted(list(repo_urls))
    except Exception as e:
        st.error(f"Error fetching existing repositories: {str(e)}")
        logger.error(f"Existing repos fetch error: {str(e)}")
        return []

# Function to generate repository description
def generate_repo_description(index, repo_url):
    try:
        if index is None:
            return "No index available to generate description."
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(f"What is the repository {repo_url} about?")
        return str(response) or "No description available."
    except Exception as e:
        st.error(f"Error generating description for {repo_url}: {str(e)}")
        logger.error(f"Description generation error: {str(e)}")
        return "Failed to generate description."

# Function to suggest questions based on repository
def suggest_questions(repo_url, lang=None):
    base_questions = [
        f"What is the main functionality of {repo_url}?",
        f"How does {repo_url} handle errors?",
        f"What are the key classes or modules in {repo_url}?",
        f"How is {repo_url} structured?",
        f"What dependencies does {repo_url} use?"
    ]
    if lang == "python":
        base_questions.append(f"How does {repo_url} handle message routing or processing?")
    elif lang in ["javascript", "typescript"]:
        base_questions.append(f"How does {repo_url} manage asynchronous operations?")
    return base_questions

# Function to process repository and return nodes
def process_repository(repo_url, github_token, branch):
    if not validate_repo_url(repo_url):
        st.error("Invalid GitHub repository URL. Use format: https://github.com/owner/repository")
        return []
    
    try:
        st.write(f"Processing {repo_url} (branch: {branch})...")
        parts = repo_url.replace("https://github.com/", "").split("/")
        owner, repository = parts[0], parts[1]
        
        github_client = GithubClient(github_token=github_token, verbose=True)
        pygithub = Github(github_token)
        repo = pygithub.get_repo(f"{owner}/{repository}")
        
        valid_branch = get_valid_branch(repo, branch)
        if not valid_branch:
            st.error(f"No valid branch found for {repo_url}. Tried '{branch}', 'main', and 'master'.")
            return []
        
        st.write(f"Fetched repo: {owner}/{repository} (branch: {valid_branch})")
        reader = GithubRepositoryReader(github_client=github_client, owner=owner, repo=repository)
        documents = reader.load_data(branch=valid_branch)
        st.write(f"Loaded {len(documents)} documents")

        all_nodes = []
        detected_languages = set()
        for doc in documents:
            filename = doc.metadata.get("filename", "")
            _, ext = os.path.splitext(filename.lower())
            lang = ext_to_lang.get(ext)
            if lang:
                detected_languages.add(lang)
            splitter = splitters.get(lang, fallback_splitter)
            nodes = splitter.get_nodes_from_documents([doc])
            for node in nodes:
                node.metadata["repo_url"] = repo_url
                node.metadata["branch"] = valid_branch
                node.metadata["node_id"] = str(uuid.uuid4())
            all_nodes.extend(nodes)
        
        st.write(f"Processed {len(all_nodes)} nodes")
        st.session_state.detected_lang = detected_languages.pop() if detected_languages else None
        return all_nodes
    except Exception as e:
        st.error(f"Error processing repository {repo_url}: {str(e)}")
        logger.error(f"Repository processing error: {str(e)}")
        return []

# Function to initialize or load Chroma database
def initialize_chroma(persist_dir, clear_db=False):
    try:
        if clear_db and os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            st.write(f"Cleared existing database at {persist_dir}")

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
def load_or_create_index(persist_dir, new_nodes=None, clear_db=False, repo_url_filter=None):
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
            if repo_url_filter:
                # Filter nodes by repo_url (not directly supported by ChromaVectorStore, so load all and filter manually)
                all_nodes = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context
                ).as_retriever().retrieve("dummy query")  # Retrieve all nodes
                filtered_nodes = [node for node in all_nodes if node.metadata.get("repo_url") == repo_url_filter]
                index = VectorStoreIndex(
                    nodes=filtered_nodes,
                    storage_context=storage_context,
                    insert_batch_size=2048
                )
            else:
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context
                )
        return index, collection
    except Exception as e:
        st.error(f"Error loading/creating index: {str(e)}")
        logger.error(f"Index creation/loading error: {str(e)}")
        return None, None

# Repository List section
st.header("Existing Repositories")
existing_repos = get_existing_repos(st.session_state.get("collection"))
if existing_repos:
    st.write("Repositories in the database:")
    for repo in existing_repos:
        st.write(f"- {repo}")
else:
    st.write("No repositories found in the database.")

# Repository Management section
st.header("Repository Management")
action = st.radio("Action", ["Add New Repository", "Load Existing Database", "Clear and Add Repository"])

if action == "Load Existing Database":
    repo_url = st.selectbox("Select Repository to Load", ["All Repositories"] + existing_repos)
    repo_url = None if repo_url == "All Repositories" else repo_url
    branch = st.text_input("Branch (optional for loading)", value="main")
else:
    repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/YonkoSam/whatsapp-python-chatbot")
    branch = st.text_input("Branch", value="main")

# Initialize session state
if 'index' not in st.session_state:
    st.session_state.index = None
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'selected_repo' not in st.session_state:
    st.session_state.selected_repo = None
if 'selected_branch' not in st.session_state:
    st.session_state.selected_branch = None
if 'repo_description' not in st.session_state:
    st.session_state.repo_description = ""
if 'detected_lang' not in st.session_state:
    st.session_state.detected_lang = None

# Handle repository selection and description
if st.button("Execute"):
    if not github_token or not google_api_key:
        st.error("Please provide GitHub Token and Google API Key in the sidebar.")
    else:
        with st.spinner("Processing..."):
            if action == "Add New Repository":
                new_nodes = process_repository(repo_url, github_token, branch)
                if new_nodes:
                    st.session_state.index, st.session_state.collection = load_or_create_index(persist_dir, new_nodes)
                    if st.session_state.index:
                        st.success(f"Repository {repo_url} added from branch {st.session_state.selected_branch or branch}! Total documents in collection: {st.session_state.collection.count()}")
                        st.session_state.selected_repo = repo_url
                        st.session_state.selected_branch = st.session_state.selected_branch or branch
                        st.session_state.repo_description = generate_repo_description(st.session_state.index, repo_url)
                    else:
                        st.error("Failed to create index.")
                else:
                    st.error("No nodes processed from repository.")
            elif action == "Load Existing Database":
                st.session_state.index, st.session_state.collection = load_or_create_index(persist_dir, repo_url_filter=repo_url)
                if st.session_state.index:
                    repo_display = repo_url if repo_url else "all repositories"
                    st.success(f"Database loaded for {repo_display}! Total documents in collection: {st.session_state.collection.count()}")
                    st.session_state.selected_repo = repo_url
                    st.session_state.selected_branch = branch
                    st.session_state.repo_description = generate_repo_description(st.session_state.index, repo_url) if repo_url else "Loaded all repositories."
                else:
                    st.error("Failed to load database.")
            else:  # Clear and Add Repository
                new_nodes = process_repository(repo_url, github_token, branch)
                if new_nodes:
                    st.session_state.index, st.session_state.collection = load_or_create_index(persist_dir, new_nodes, clear_db=True)
                    if st.session_state.index:
                        st.success(f"Database cleared and repository {repo_url} added from branch {st.session_state.selected_branch or branch}! Total documents in collection: {st.session_state.collection.count()}")
                        st.session_state.selected_repo = repo_url
                        st.session_state.selected_branch = st.session_state.selected_branch or branch
                        st.session_state.repo_description = generate_repo_description(st.session_state.index, repo_url)
                    else:
                        st.error("Failed to create index.")
                else:
                    st.error("No nodes processed from repository.")

# Display repository description and question suggestions
if st.session_state.selected_repo:
    st.subheader(f"Selected Repository: {st.session_state.selected_repo} (Branch: {st.session_state.selected_branch})")
    st.write("**Description:**")
    st.write(st.session_state.repo_description)
    
    st.write("**Suggested Questions:**")
    suggested_questions = suggest_questions(st.session_state.selected_repo, st.session_state.detected_lang)
    selected_question = st.selectbox("Choose a question or type your own:", [""] + suggested_questions)
    if selected_question:
        st.session_state.query = selected_question

# Query section
st.header("Ask Questions")
query = st.text_input("Enter your question", value=st.session_state.get("query", "How does the WhatsApp bot handle message routing?"))
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