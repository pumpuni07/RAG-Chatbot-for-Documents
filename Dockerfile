# ── Dockerfile ────────────────────────────────────────────────────────────────
# PDF Chatbot — IBM Generative AI Engineering Professional Certificate
# Author: Jack Pumpuni Frimpong-Manso | 2026
# ──────────────────────────────────────────────────────────────────────────────

# Use official Python 3.10 slim image (smaller footprint)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (layer caching — only reinstalls on changes)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir \
    langchain==0.3.27 \
    langchain-core==0.3.80 \
    langchain-community==0.3.31 \
    langchain-huggingface==0.3.1 \
    langchain-text-splitters==0.3.11 \
    huggingface-hub==0.36.0 \
    tokenizers==0.19.1 \
    transformers==4.43.3 \
    sentence-transformers==2.7.0 \
    requests==2.32.5

# Copy the rest of the application
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 8000

# Run the Flask server
CMD ["python", "-u", "server.py"]
