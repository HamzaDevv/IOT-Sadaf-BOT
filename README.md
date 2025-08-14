# Sadaf Conversational AI Assistant

A robust, modular, and extensible conversational AI assistant with human-like long-term memory, multimodal (vision) capabilities, and tool integration (e.g., Google Maps). Built with Python, LLMs (Ollama, Gemini), ChromaDB, and modern async speech interfaces.

---

## Architecture Overview

**Key Components:**

- **main.py**: Entry point. Handles the async conversation loop, speech I/O, and interaction with the memory system.
- **memory/**: Modular memory system.
  - `conversational_memory.py`: Manages buffer, summarization, and long-term memory storage/retrieval.
  - `llm_summarizer.py`: Summarizes conversation into structured facts using LLMs.
  - `knowledge_base.py`: Handles persistent vector storage (ChromaDB) and embedding via Ollama.
  - `pydantic_model.py`: Strict Pydantic schemas for context and fact extraction.
- **tools/**: (Planned) Tool integrations for vision (image-to-Gemini) and Google Maps API.
- **speak.py / listen.py**: Async macOS TTS and speech recognition for natural voice interaction.
- **personality.py**: Dynamic system prompt for different AI personalities.
- **config.py**: Centralized configuration for all tunable parameters.

**Data Flow:**
1. User speaks → audio transcribed → text passed to main loop.
2. ConversationManager buffers turns, summarizes with LLM, and stores only high-value facts in ChromaDB.
3. On each turn, context is built from recent buffer, summaries, and relevant long-term memory.
4. LLM generates a response, which is spoken back to the user.
5. (Optional) User can upload images or request map info via tools (planned/extendable).

---

## Features
- **Human-like memory**: Stores only explicit, important facts; deduplicates and scores for worthiness.
- **Multimodal**: (Planned) Accepts images, queries Gemini Vision for image understanding.
- **Tool Augmentation**: (Planned) Google Maps API for location-based queries.
- **Speech-enabled**: Async TTS and speech recognition for hands-free use.
- **Configurable**: All parameters in `config.py` with clear explanations.
- **Testable**: `test.py` for simulating conversations and inspecting memory/context.

---

## Setup & Usage

### 1. **Install Requirements**

```sh
pip install -r requirements.txt
```

- Ensure you have [Ollama](https://ollama.com/) running locally for embeddings and LLM inference.
- (Optional) Set up Gemini API and Google Maps API for tool integrations.

### 2. **Configure**
- Edit `config.py` to set model names, buffer sizes, API keys, etc.

### 3. **Run the Assistant**

```sh
python3 main.py
```
- Speak to the assistant; it will respond and build memory over time.

### 4. **Test Memory System**

```sh
python3 test.py
```
- Runs sample conversations and prints what the system stores as context/summary.

### 5. **(Planned) Use Tools**
- Place tool scripts in `tools/` (e.g., for image upload or map queries).
- Extend `main.py` and `ConversationManager` to call these tools as needed.

---

## File/Folder Guide

- `main.py` — Main async loop, speech I/O, and conversation logic
- `memory/` — All memory, summarization, and vector DB logic
- `tools/` — (Planned) Tool integrations (vision, maps, etc.)
- `config.py` — All configuration
- `test.py` — Test harness for memory/context
- `chroma_db/` — Persistent vector DB storage
- `transcriptions.txt`, `conversation_summary.txt`, `stored_facts.txt` — Logs and memory outputs

---

## Extending the Project
- Add new tools in `tools/` and register them in `main.py`.
- To add vision: write a tool that takes an image, sends it to Gemini Vision, and returns the result.
- To add maps: write a tool that queries Google Maps API and returns info to the user.
- Update `ConversationManager` to call tools based on user intent.

---

## Credits
- Built by Ameer Hamza Khan
- Uses open-source LLMs, ChromaDB, and modern Python async libraries.

---

## License
MIT License (see LICENSE file)
