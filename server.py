"""
server.py — Flask Backend Server
=================================
IBM Generative AI Engineering Professional Certificate
Project: Build a Chatbot for Your Data

Author  : Jack Pumpuni Frimpong-Manso
Date    : 2026
License : Apache 2.0

Description:
    Flask web server that exposes three routes:
      GET  /                  → serves the chat interface (index.html)
      POST /process-document  → receives a PDF, saves it, triggers RAG indexing
      POST /process-message   → receives a user message, returns bot response

    On startup, calls worker.init_llm() to initialise the LLM and embeddings
    so they are ready before the first request arrives.
"""

import os
import base64
import logging

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import worker

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow cross-origin requests (required for some browser environments)

# ── Upload folder ─────────────────────────────────────────────────────────────
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ── Initialise LLM on startup ─────────────────────────────────────────────────
logger.info("Initialising LLM and embeddings at startup...")
worker.init_llm()
logger.info("LLM and embeddings ready.")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Serve the chatbot frontend interface."""
    return render_template("index.html")


@app.route("/process-document", methods=["POST"])
def process_document_route():
    """
    Receive a base64-encoded PDF from the frontend, decode and save it,
    then trigger the RAG indexing pipeline in worker.py.

    Expected JSON body
    ------------------
    {
        "fileData": "data:application/pdf;base64,<base64string>"
    }

    Returns
    -------
    JSON: { "botResponse": "Thank you! ..." }
    """
    try:
        data = request.get_json()
        if not data or "fileData" not in data:
            logger.warning("No fileData received in /process-document request.")
            return jsonify({"botResponse": "No document data received. Please try again."}), 400

        file_data = data["fileData"]

        # Strip the data-URI prefix if present (e.g. "data:application/pdf;base64,")
        if "," in file_data:
            file_data = file_data.split(",", 1)[1]

        # Decode and save the PDF
        pdf_bytes = base64.b64decode(file_data)
        document_path = os.path.join(UPLOAD_FOLDER, "uploaded_document.pdf")
        with open(document_path, "wb") as f:
            f.write(pdf_bytes)
        logger.info("PDF saved to: %s (%d bytes)", document_path, len(pdf_bytes))

        # Trigger RAG indexing
        worker.process_document(document_path)

        return jsonify({
            "botResponse": (
                "Document uploaded and processed successfully! "
                "You can now ask me questions about its content."
            )
        })

    except Exception as exc:
        logger.error("Error processing document: %s", exc, exc_info=True)
        return jsonify({"botResponse": f"Error processing document: {str(exc)}"}), 500


@app.route("/process-message", methods=["POST"])
def process_message_route():
    """
    Receive a user message and return the chatbot's grounded response.

    Expected JSON body
    ------------------
    {
        "userMessage": "What is the main topic of this document?"
    }

    Returns
    -------
    JSON: { "botResponse": "<answer from LLM>" }
    """
    try:
        data = request.get_json()
        if not data or "userMessage" not in data:
            logger.warning("No userMessage received in /process-message request.")
            return jsonify({"botResponse": "No message received. Please try again."}), 400

        user_message = data["userMessage"].strip()
        logger.info("Received user message: '%s'", user_message)

        if not user_message:
            return jsonify({"botResponse": "Please enter a question."})

        if worker.conversation_retrieval_chain is None:
            return jsonify({
                "botResponse": (
                    "Please upload a PDF document first before asking questions."
                )
            })

        # Get response from RAG pipeline
        bot_response = worker.process_prompt(user_message)

        return jsonify({"botResponse": bot_response})

    except Exception as exc:
        logger.error("Error processing message: %s", exc, exc_info=True)
        return jsonify({"botResponse": f"Error generating response: {str(exc)}"}), 500


@app.route("/reset", methods=["POST"])
def reset_route():
    """
    Reset the chat history so a new conversation can begin.
    Called when the user clicks the reset button in the frontend.
    """
    worker.chat_history = []
    logger.info("Chat history reset.")
    return jsonify({"botResponse": "Chat history cleared. Ready for a new conversation!"})


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8000, host="0.0.0.0")
