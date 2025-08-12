from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from typing import Optional
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ------------------ AI Assistant Config ------------------
AI_NAME = "Sadaf"  # Name of the AI assistant

# ------------------ LLM Config ------------------
TEMPERATURE_OF_LLM = 0.0  # 0.0 = deterministic output, 1.0 = more creative
GLOBAL_LLM = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", temperature=TEMPERATURE_OF_LLM
)
OLLAMA_LLM = ChatOllama(model="llama3.2", base_url="http://localhost:11434")

# ------------------ Embedding Config ------------------
OLLAMA_EMBED_URL = (
    "http://localhost:11434/api/embeddings"  # Ollama embeddings API endpoint
)
OLLAMA_EMBED_MODEL = "nomic-embed-text"  # Embedding model for vector search

# ------------------ Memory Config ------------------
BUFFER_SIZE = 10  # Max number of recent interactions to keep in short-term buffer
SUMMARY_SIZE = 5  # Number of conversations to summarize into mid-term memory
CHROMA_DB_DIR = "./chroma_db"  # Directory path for Chroma vector DB storage
MIN_IMPORTANCE_TO_STORE = 0.0  # Minimum importance score before storing memory
N_FACTS_CONTEXT = 3  # Number of facts to retrieve for context in responses
SESSION_METADATA_FILE = "session_metadata.json"  # File for saving session metadata

# ------------------ TTS/Voice Config ------------------
VOICE = "Samantha"  # Voice name for text-to-speech
MAX_RESPONSE_WORDS = 70  # Maximum number of words per spoken response

# ------------------ Audio Recording Config ------------------
TIMEOUT = 4  # Max seconds to wait for speech to start after listening begins
PHRASE_TIME_LIMIT = 30  # Max duration (seconds) for one continuous speech segment
OUTPUT_FILE = "transcriptions.txt"  # File to save speech-to-text transcripts
RECORDING_TIME_IN_SECONDS = 900  # Max continuous recording session time in seconds

# ------------------ Speech Recognition Tuning ------------------
PAUSE_THRESHOLD = 2.5  # Seconds of silence before considering the phrase complete
ENERGY_THRESHOLD = 300  # Minimum volume level to consider sound as speech (vs. noise)
DYNAMIC_ENERGY_THRESHOLD = (
    True  # Whether to auto-adjust ENERGY_THRESHOLD to background noise
)
_NOISE_PROFILE: Optional[np.ndarray] = (
    None  # Stores pre-computed background noise profile for reuse
)
