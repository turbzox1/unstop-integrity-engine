"""Pre-download the open models used by the pipeline (run once, needs internet)."""
from sentence_transformers import SentenceTransformer
import whisper

SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
whisper.load_model("base")
print("Models cached: all-MiniLM-L6-v2, whisper-base")
