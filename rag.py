import os

from llama_index.readers.github import (
    GithubRepositoryReader,
    GithubClient,
)

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.settings import Settings

from llama_index.core.node_parser import CodeSplitter, SimpleNodeParser

from dotenv import load_dotenv
load_dotenv()


google_api_key = os.environ["GOOGLE_API_KEY"]

Settings.llm = GoogleGenAI(model=os.environ["GOOGLE_MODEL"], temperature=0.1, api_key=google_api_key)
Settings.embed_model = GoogleGenAIEmbedding(model=os.environ["GOOGLE_EMBEDDING_MODEL"], api_key=google_api_key)


url = "https://github.com/YonkoSam/whatsapp-python-chatbot"

parts = url.replace("https://github.com/", "").split("/")

owner = parts[0]
repository = parts[1]


github_client = GithubClient(
   github_token=os.environ["GITHUB_TOKEN"],
   verbose=True
)


from github import Github 

pygithub = Github(os.environ["GITHUB_TOKEN"])
repo = pygithub.get_repo(f"{owner}/{repository}")
latest_commit_sha = repo.get_branch("main").commit.sha

reader = GithubRepositoryReader(
   github_client=github_client,
   owner=owner,
   repo=repository,
)

documents = reader.load_data(branch="main")



# Languages you want to support (just examples from your list)
languages = [
    "python", "javascript", "java", "c", "cpp", "php", "ruby", "rust",
    "go", "typescript", "swift", "kotlin", "scala", "bash", "sql", "html", "css",
    "lua", "elisp", "markdown", "yaml", "json", "perl", "r", "objc", "haskell",
    "toml", "fortran", "hack", "commonlisp",
]

# Create splitters per language
splitters = {
    lang: CodeSplitter(language=lang, chunk_lines=40, chunk_lines_overlap=15, max_chars=1500)
    for lang in languages
}

# Fallback generic parser for unsupported languages
fallback_splitter = SimpleNodeParser()


ext_to_lang = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".php": "php",
    ".rb": "ruby",
    ".rs": "rust",
    ".go": "go",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".lua": "lua",
    ".el": "elisp",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".pl": "perl",
    ".r": "r",
    ".m": "objc",      # careful: also Matlab has .m
    ".hs": "haskell",
    ".toml": "toml",
    ".f90": "fortran",
    ".f": "fortran",
    ".hack": "hack",
    ".lisp": "commonlisp",
    # add more as needed
}

all_nodes = []
for doc in documents:
    filename = doc.metadata.get("filename", "")
    _, ext = os.path.splitext(filename.lower())

    lang = ext_to_lang.get(ext)
    if lang in splitters:
        splitter = splitters[lang]
    else:
        splitter = fallback_splitter

    nodes = splitter.get_nodes_from_documents([doc])
    all_nodes.extend(nodes)

print(f"Total nodes created: {len(all_nodes)}")


from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import shutil
import os

# Delete existing chroma db directory if needed (optional clean start)
persist_dir = "./chroma_db"
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

# Set up Chroma client and collection
chroma_client = chromadb.PersistentClient(path=persist_dir)

index_name = "MyExternalContext"
collection = chroma_client.get_or_create_collection(name=index_name)

# Construct vector store
vector_store = ChromaVectorStore(
    chroma_collection=collection,
)

# Set up the storage context with Chroma vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Assume `nodes` already contains your list of chunked documents
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context
)

# Optional: print collection stats
print(f"Collection name: {collection.name}")
print(f"Number of documents stored: {collection.count()}")

from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank

# Replace chunk text with the `window` metadata (from node parser)
metadata_postproc = MetadataReplacementPostProcessor(
    target_metadata_key="window"  # Default key used in CodeSplitter
)

# Re-rank top results using a transformer model
reranker = SentenceTransformerRerank(
    top_n=2,
    model="BAAI/bge-reranker-base"
)

# Get the query engine
query_engine = index.as_query_engine(
    similarity_top_k=5,
    node_postprocessors=[
        metadata_postproc,
        reranker
    ]
)

response = query_engine.query("How does the WhatsApp bot handle message routing?")
print(response)



