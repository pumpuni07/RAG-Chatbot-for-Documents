# 📄 PDF Chatbot — RAG-Based Document Q&A System

> **IBM Generative AI Engineering Professional Certificate**
> Project: *Build a Chatbot for Your Data*
> **Author: Jack Pumpuni Frimpong-Manso**

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.27-green)](https://www.langchain.com/)
[![IBM WatsonX](https://img.shields.io/badge/IBM-WatsonX-054ADA?logo=ibm)](https://www.ibm.com/watsonx)
[![Flask](https://img.shields.io/badge/Flask-2.3-black?logo=flask)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)

---

## 📌 Overview

A full-stack conversational AI application that allows users to upload any PDF document and ask natural language questions about its content. Built on a **Retrieval-Augmented Generation (RAG)** pipeline that grounds every answer in the actual document — eliminating hallucination.

**Key technologies:** IBM WatsonX (Llama 4 Maverick), LangChain, HuggingFace Embeddings, Chroma Vector Store, Flask, Docker.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                             │
│   Upload PDF  ──────────────────────────►  Ask Question         │
└──────────┬──────────────────────────────────────┬──────────────┘
           │ POST /process-document               │ POST /process-message
           ▼                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Flask Backend (server.py)                     │
└──────────┬──────────────────────────────────────┬──────────────┘
           │                                      │
           ▼                                      ▼
┌──────────────────────┐              ┌───────────────────────────┐
│  process_document()  │              │    process_prompt()        │
│                      │              │                            │
│  PyPDFLoader         │              │  RetrievalQA.invoke()      │
│       │              │              │         │                  │
│  RecursiveCharacter  │              │  Chroma MMR Retrieval      │
│  TextSplitter        │              │  (k=6, λ=0.25)            │
│  (1024, overlap=64)  │              │         │                  │
│       │              │              │  WatsonxLLM (Llama 4)     │
│  HuggingFace         │              │         │                  │
│  Embeddings          │              │  Grounded Answer           │
│  (MiniLM-L6-v2)      │              │                            │
│       │              │              │  chat_history updated      │
│  Chroma Vector DB ───┼──────────────┘                            │
└──────────────────────┘                                           │
```

### Why RAG?

Standard LLMs have a fixed knowledge cutoff and no access to private documents. RAG solves this by:
1. **Retrieving** the most relevant chunks from your document at query time
2. **Augmenting** the LLM prompt with those chunks as context
3. **Generating** an answer grounded in your actual document content

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📤 PDF Upload | Drag and drop or click to upload any PDF |
| 💬 Conversational memory | Bot remembers previous questions in the session |
| 🔍 MMR Retrieval | Maximal Marginal Relevance for diverse, non-redundant context |
| 🌗 Light / Dark mode | Toggle in the navbar |
| 🔄 Chat reset | Clear history and start fresh |
| 🐳 Docker support | Containerised for consistent deployment |
| 🔒 Input sanitisation | XSS-safe frontend input cleaning |
| 📱 Responsive | Works on desktop and mobile |

---

## 📁 Project Structure

```
build_chatbot_for_your_data/
│
├── server.py                  # Flask backend — routes & request handling
├── worker.py                  # Core RAG pipeline — LLM, embeddings, retrieval
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container definition
├── .gitignore
│
├── templates/
│   └── index.html             # Chat interface
│
├── static/
│   ├── style.css              # Interface styling + animations
│   └── script.js              # Frontend logic
│
└── uploads/                   # Temporary PDF storage (git-ignored)
```

---

## ⚙️ Setup & Installation

### Option A — Local (virtual environment)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/build_chatbot_for_your_data.git
cd build_chatbot_for_your_data

# 2. Create and activate virtual environment
pip3 install virtualenv
virtualenv my_env
source my_env/bin/activate        # Linux / macOS
# my_env\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
python3 server.py
```

Open your browser at `http://127.0.0.1:8000`

---

### Option B — Docker

```bash
# Build image
docker build --no-cache -t build_chatbot_for_your_data .

# Run container
docker run -p 8000:8000 build_chatbot_for_your_data
```

Open your browser at `http://localhost:8000`

---

## 🔑 Credentials

### Inside IBM Skills Network Cloud IDE
No credentials needed — `project_id="skills-network"` is handled automatically.

### Outside IBM Skills Network (local / production)

Create a `.env` file in the project root:

```env
WATSONX_API_KEY=your-watsonx-api-key
WATSONX_PROJECT_ID=your-project-id
HUGGINGFACEHUB_API_TOKEN=your-huggingface-token
```

Then update `worker.py`:
```python
import os
from dotenv import load_dotenv
load_dotenv()

Watsonx_API = os.getenv("WATSONX_API_KEY")
Project_id  = os.getenv("WATSONX_PROJECT_ID")
```

---

## 🧠 Key Implementation Details

### `worker.py`

#### `init_llm()`
Initialises two global components used across the session:
- **WatsonxLLM** — `meta-llama/llama-4-maverick-17b-128e-instruct-fp8` with `max_new_tokens=256`, `temperature=0.1`
- **HuggingFaceEmbeddings** — `sentence-transformers/all-MiniLM-L6-v2` (384-dim, optimised for semantic similarity)

#### `process_document(document_path)`
| Step | Tool | Detail |
|---|---|---|
| Load | `PyPDFLoader` | Reads PDF page by page |
| Split | `RecursiveCharacterTextSplitter` | chunk_size=1024, overlap=64 |
| Embed | `HuggingFaceEmbeddings` | 384-dim vectors per chunk |
| Index | `Chroma` | In-memory vector store |
| Chain | `RetrievalQA` | MMR retrieval, k=6, λ=0.25 |

#### `process_prompt(prompt)`
- Passes `prompt` + `chat_history` to `RetrievalQA.invoke()`
- Retrieves top-6 diverse chunks via MMR
- Generates grounded answer via WatsonX LLM
- Appends `(prompt, answer)` tuple to `chat_history` for multi-turn memory

### `server.py` — Flask Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serve chat interface |
| `/process-document` | POST | Receive PDF, trigger RAG indexing |
| `/process-message` | POST | Receive question, return answer |
| `/reset` | POST | Clear chat history |

---

## 🎓 Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Retrieval-Augmented Generation | LangChain `RetrievalQA` + Chroma |
| Vector embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Semantic search | Chroma MMR retrieval |
| LLM integration | IBM WatsonX `WatsonxLLM` (Llama 4) |
| Conversational memory | `chat_history` tuple list |
| REST API design | Flask routes |
| Containerisation | Docker |
| Text chunking | Recursive character splitting with overlap |
| Frontend-backend integration | Fetch API + Flask JSON responses |
| Input sanitisation | XSS-safe JS cleaning |

---

## 🔗 Related Portfolio Projects

- [SpaceX Falcon 9 Landing Prediction](../spacex-falcon9-ml) — IBM Applied Data Science Capstone
- [Dissolved Inorganic Nutrient Prediction](../nutrient-ml-pipeline) — Extra Trees Regression, R²>0.96
- [Watson NLP Emotion Detection](../emotion-detection-nlp) — IBM AI Engineering Capstone

---

## 📜 License

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
Original lab framework © IBM Skills Network. Implementation by Jack Pumpuni Frimpong-Manso.

---

## 🙏 Acknowledgements

- IBM Skills Network — course framework and WatsonX access
- Meta AI — Llama 4 Maverick model
- HuggingFace — `sentence-transformers` library
- LangChain — RAG orchestration framework
