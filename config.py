import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"                 # small = fast on CPU
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # shared characters between neighbouring chunks
TOP_K_RETRIEVE = 20    # candidates taken from each search
TOP_K_FINAL = 5        # chunks sent to the LLM
RRF_K = 60

HISTORY_TURNS = 4      # how many past messages are used to rewrite follow-ups

DB_DIR = "data/chroma"
PAPERS_DIR = "data/papers"
EVAL_FILE = "data/eval_set.json"
