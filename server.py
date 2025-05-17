import os
import shutil
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.settings import Settings
from llama_index.core.node_parser import CodeSplitter, SimpleNodeParser
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

try:
    logger.info("Loading environment variables and initializing models...")
    google_api_key = os.environ["GOOGLE_API_KEY"]

    Settings.llm = GoogleGenAI(model=os.environ["GOOGLE_MODEL"], temperature=0.1, api_key=google_api_key)
    Settings.embed_model = GoogleGenAIEmbedding(model=os.environ["GOOGLE_EMBEDDING_MODEL"], api_key=google_api_key)
    logger.info("Model and embedding initialized successfully.")

    persist_dir = "./chroma_db"
    logger.info(f"Using ChromaDB persistence directory: {persist_dir}")

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    index_name = "MyExternalContext"
    collection = chroma_client.get_or_create_collection(name=index_name)
    logger.info(f"Loaded Chroma collection: {collection.name}, documents count: {collection.count()}")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load your pre-built index from storage (replace [] with actual nodes if rebuilding)
    index = VectorStoreIndex([], storage_context=storage_context)
    logger.info("VectorStoreIndex initialized.")

except Exception as e:
    logger.error(f"Error during initialization: {e}")
    raise e  # Prevent server from starting if initialization fails

class QueryRequest(BaseModel):
    query: str
    similarity_top_k: Optional[int] = 5

@app.post("/query")
def query(request: QueryRequest):
    logger.info(f"Received query: {request.query} with top_k={request.similarity_top_k}")
    try:
        query_engine = index.as_query_engine(similarity_top_k=request.similarity_top_k)
        response = query_engine.query(request.query)
        logger.debug(f"Query response: {response}")
        return {"response": str(response)}
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
