"""
worker.py — Document Processing and Conversation Management
============================================================
IBM Generative AI Engineering Professional Certificate
Project: Build a Chatbot for Your Data (RAG-based PDF Chatbot)

Author  : Jack Pumpuni Frimpong-Manso
Date    : 2026
License : Apache 2.0

Description:
    Core AI logic of the chatbot. Initialises the WatsonX LLM (Llama 4
    via IBM watsonX), sets up HuggingFace sentence embeddings, processes
    uploaded PDF documents into a Chroma vector store, and manages a
    conversational retrieval chain (RAG pipeline) that answers user
    questions grounded in the document content.

RAG Pipeline:
    PDF Document
        └─► PyPDFLoader              (load pages)
                └─► RecursiveCharacterTextSplitter   (chunk text)
                        └─► HuggingFaceEmbeddings    (vectorise chunks)
                                └─► Chroma DB         (store & retrieve)
                                        └─► RetrievalQA Chain
                                                └─► WatsonxLLM (Llama 4)
                                                        └─► Answer
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import logging

# ── LangChain ─────────────────────────────────────────────────────────────────
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ibm import WatsonxLLM

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if os.environ.get("USE_GPU", "0") == "1" else "cpu"
logger.info("Running on device: %s", DEVICE)

# ── Credentials (replace when running outside IBM Skills Network IDE) ─────────
Watsonx_API = "Your WatsonX API"
Project_id  = "Your Project ID"

# ── Global state ──────────────────────────────────────────────────────────────
llm_hub                      = None
embeddings                   = None
conversation_retrieval_chain = None
chat_history                 = []


# ─────────────────────────────────────────────────────────────────────────────
# 1. INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def init_llm():
    """
    Initialise the WatsonX LLM and HuggingFace sentence embeddings.

    Called once at application startup from server.py.
    Sets global ``llm_hub`` and ``embeddings``.

    LLM
    ---
    model : meta-llama/llama-4-maverick-17b-128e-instruct-fp8
    temp  : 0.1  — near-deterministic, focused answers
    tokens: 256  — caps response length

    Embeddings
    ----------
    model : sentence-transformers/all-MiniLM-L6-v2
    dims  : 384-dimensional vectors
    why   : Fast, lightweight, strong semantic similarity for doc Q&A
    """
    global llm_hub, embeddings

    logger.info("Initialising WatsonxLLM and HuggingFace embeddings...")

    # ── WatsonX LLM ───────────────────────────────────────────────────────────
    MODEL_ID    = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
    PROJECT_ID  = "skills-network"  # replace with your own outside SN IDE

    model_parameters = {
        "max_new_tokens": 256,
        "temperature": 0.1,
    }

    llm_hub = WatsonxLLM(
        model_id=MODEL_ID,
        url=WATSONX_URL,
        project_id=PROJECT_ID,
        params=model_parameters,
    )
    logger.debug("WatsonxLLM initialised: %s", llm_hub)

    # ── HuggingFace Embeddings ────────────────────────────────────────────────
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE},
    )
    logger.debug("Embeddings initialised on device: %s", DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DOCUMENT PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_document(document_path):
    """
    Load, chunk, embed, and index a PDF document for retrieval.

    Parameters
    ----------
    document_path : str
        Path to the PDF file to process.

    Pipeline
    --------
    1. Load  — PyPDFLoader reads the PDF page-by-page.
    2. Split — RecursiveCharacterTextSplitter creates overlapping chunks
               (chunk_size=1024, overlap=64) so context is not lost at
               chunk boundaries.
    3. Embed — HuggingFaceEmbeddings converts each chunk to a vector.
    4. Index — Chroma stores vectors in an in-memory vector store.
    5. Chain — RetrievalQA wraps LLM + retriever into a callable chain.

    Retrieval strategy
    ------------------
    MMR (Maximal Marginal Relevance) with lambda_mult=0.25 balances
    relevance and diversity, reducing redundant chunks in the LLM context.
    """
    global conversation_retrieval_chain

    # Step 1: Load PDF
    logger.info("Loading document: %s", document_path)
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    logger.debug("Loaded %d page(s).", len(documents))

    # Step 2: Split into overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=64,
    )
    texts = text_splitter.split_documents(documents)
    logger.debug("Split into %d chunk(s).", len(texts))

    # Step 3 & 4: Embed and store in Chroma
    logger.info("Building Chroma vector store...")
    db = Chroma.from_documents(texts, embedding=embeddings)
    logger.debug("Chroma vector store ready.")

    try:
        collections = db._client.list_collections()
        logger.debug("Chroma collections: %s", collections)
    except Exception as exc:
        logger.warning("Could not list Chroma collections: %s", exc)

    # Step 5: Build RetrievalQA chain
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "lambda_mult": 0.25},
        ),
        return_source_documents=False,
        input_key="question",
    )
    logger.info("RetrievalQA chain created successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROMPT PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_prompt(prompt):
    """
    Process a user question and return a grounded answer from the document.

    Parameters
    ----------
    prompt : str
        The user's question or message.

    Returns
    -------
    str
        LLM answer grounded in the processed PDF content.

    How it works
    ------------
    1. Passes prompt + chat_history to the RetrievalQA chain.
    2. Chain retrieves top-6 diverse chunks from Chroma.
    3. LLM generates an answer grounded in those chunks.
    4. (prompt, answer) tuple appended to chat_history for multi-turn memory.

    Notes
    -----
    chat_history is a list of (human_message, ai_message) tuples —
    the format expected by LangChain conversational chains.
    """
    global conversation_retrieval_chain
    global chat_history

    logger.info("Processing prompt: '%s'", prompt)

    # Query the RAG chain
    output = conversation_retrieval_chain.invoke(
        {
            "question": prompt,
            "chat_history": chat_history,
        }
    )
    answer = output["result"]
    logger.debug("Model answer: %s", answer)

    # Update conversational memory
    chat_history.append((prompt, answer))
    logger.debug("Chat history updated. Total exchanges: %d", len(chat_history))

    return answer
