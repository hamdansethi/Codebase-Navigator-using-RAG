# ================================================================
# 1. IMPORTS AND ENVIRONMENT SETUP
# ================================================================

import os
import shutil
import json
import logging
from dotenv import load_dotenv

import chromadb
import requests

from flask import Flask, request, jsonify
from github import Github

from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.settings import Settings
from llama_index.core.node_parser import CodeSplitter, SimpleNodeParser
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank

# ================================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ================================================================

load_dotenv()
google_api_key = os.environ["GOOGLE_API_KEY"]

# ================================================================
# 3. SETUP LLM AND EMBEDDING MODEL
# ================================================================

Settings.llm = GoogleGenAI(
    model=os.environ["GOOGLE_MODEL"],
    temperature=0.1,
    api_key=google_api_key
)

Settings.embed_model = GoogleGenAIEmbedding(
    model=os.environ["GOOGLE_EMBEDDING_MODEL"],
    api_key=google_api_key
)

# ================================================================
# 4. GITHUB REPO CONFIGURATION
# ================================================================

url = "https://github.com/YonkoSam/whatsapp-python-chatbot"
owner, repository = url.replace("https://github.com/", "").split("/")

# Set up GitHub client
github_client = GithubClient(
    github_token=os.environ["GITHUB_TOKEN"],
    verbose=True
)

pygithub = Github(os.environ["GITHUB_TOKEN"])
repo = pygithub.get_repo(f"{owner}/{repository}")
latest_commit_sha = repo.get_branch("main").commit.sha

# ================================================================
# 5. LOAD DOCUMENTS FROM GITHUB REPOSITORY
# ================================================================

reader = GithubRepositoryReader(
    github_client=github_client,
    owner=owner,
    repo=repository
)

documents = reader.load_data(branch="main")

# ================================================================
# 6. SPLIT DOCUMENTS INTO CHUNKS BY LANGUAGE
# ================================================================

languages = [
    "python", "javascript", "java", "c", "cpp", "php", "ruby", "rust",
    "go", "typescript", "swift", "kotlin", "scala", "bash", "sql", "html", "css",
    "lua", "elisp", "markdown", "yaml", "json", "perl", "r", "objc", "haskell",
    "toml", "fortran", "hack", "commonlisp"
]

splitters = {
    lang: CodeSplitter(language=lang, chunk_lines=40, chunk_lines_overlap=15, max_chars=1500)
    for lang in languages
}
fallback_splitter = SimpleNodeParser()

ext_to_lang = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".java": "java", ".c": "c",
    ".cpp": "cpp", ".php": "php", ".rb": "ruby", ".rs": "rust", ".go": "go", ".swift": "swift",
    ".kt": "kotlin", ".scala": "scala", ".sh": "bash", ".sql": "sql", ".html": "html", ".css": "css",
    ".lua": "lua", ".el": "elisp", ".md": "markdown", ".yaml": "yaml", ".yml": "yaml", ".json": "json",
    ".pl": "perl", ".r": "r", ".m": "objc", ".hs": "haskell", ".toml": "toml", ".f90": "fortran",
    ".f": "fortran", ".hack": "hack", ".lisp": "commonlisp"
}

all_nodes = []
for doc in documents:
    filename = doc.metadata.get("filename", "")
    _, ext = os.path.splitext(filename.lower())
    lang = ext_to_lang.get(ext)
    splitter = splitters.get(lang, fallback_splitter)
    nodes = splitter.get_nodes_from_documents([doc])
    all_nodes.extend(nodes)

print(f"Total nodes created: {len(all_nodes)}")

# ================================================================
# 7. VECTOR STORE SETUP USING CHROMA DB
# ================================================================

persist_dir = "./chroma_db"
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

chroma_client = chromadb.PersistentClient(path=persist_dir)
collection = chroma_client.get_or_create_collection(name="MyExternalContext")

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    all_nodes,
    storage_context=storage_context
)

print(f"Collection name: {collection.name}")
print(f"Number of documents stored: {collection.count()}")

# ================================================================
# 8. POST-PROCESSORS FOR QUERY ENHANCEMENT
# ================================================================

metadata_postproc = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)

reranker = SentenceTransformerRerank(
    top_n=2,
    model="BAAI/bge-reranker-base"
)

# ================================================================
# 9. BUILD QUERY ENGINE AND MAKE A TEST QUERY
# ================================================================

query_engine = index.as_query_engine(
    similarity_top_k=5,
    node_postprocessors=[metadata_postproc, reranker]
)

response = query_engine.query("How does the WhatsApp bot handle message routing?")
print(response)
