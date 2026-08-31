# src/config.py

import os
from dotenv import load_dotenv

load_dotenv()  

# --- Weaviate ---
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", 8090))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))
PRODUCTS_COLLECTION_NAME = os.getenv("PRODUCTS_COLLECTION_NAME", "products")

# --- Ollama ---
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://host.docker.internal:11434")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
INTENT_CLASSIFICATION_MODEL = os.getenv("INTENT_CLASSIFICATION_MODEL", "deepseek-r1:7b")  
METADATA_EXTRACTION_MODEL = os.getenv("METADATA_EXTRACTION_MODEL", "llama3.2")             
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "llama3.2")

# --- Retrieval ---
DEFAULT_SEARCH_LIMIT = int(os.getenv("DEFAULT_SEARCH_LIMIT", 20))
MIN_RESULTS_THRESHOLD = int(os.getenv("MIN_RESULTS_THRESHOLD", 10))
FALLBACK_RESULTS_THRESHOLD = int(os.getenv("FALLBACK_RESULTS_THRESHOLD", 5))

FILTER_IMPORTANCE_ORDER = [
    "gender",
    "masterCategory",
    "articleType",
    "price",
    "baseColour",
    "season",
    "usage",
]

# --- Generation parameters per task type ---
GENERATION_PARAMS = {
    "creative": {"top_p": 0.7, "temperature": 1.2},
    "technical": {"top_p": 0.9, "temperature": 0.1},
}

# --- Chat memory ---
MAX_CHAT_MEMORY_TURNS = int(os.getenv("MAX_CHAT_MEMORY_TURNS", 6))

# --- Data paths ---
DATA_DIR = os.getenv("DATA_DIR", "data")
PRODUCTS_DATA_PATH = os.path.join(DATA_DIR, "clothes_json.joblib")
FAQ_DATA_PATH = os.path.join(DATA_DIR, "faq.joblib")