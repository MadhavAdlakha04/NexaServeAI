# NexaServeAI — RAG Customer Support Chatbot

NexaServeAI is a Retrieval-Augmented Generation (RAG) based customer support chatbot designed to answer enterprise support queries using internal documentation.

The system retrieves relevant context from company documents using vector search and generates accurate responses using a large language model.

---

## Overview

Traditional chatbots often hallucinate when answering questions. NexaServeAI solves this by retrieving relevant information from a knowledge base before generating responses.

Pipeline:

User Question → Vector Search → Retrieve Context → LLM Response

---

## Architecture

1. User submits a query
2. Query is converted into embeddings
3. FAISS performs vector similarity search
4. Relevant document chunks are retrieved
5. Context + question sent to LLM
6. Generated answer returned to user

---

## Tech Stack

Python
LangChain
FAISS Vector Database
Mistral-7B LLM
BAAI BGE Embeddings
HuggingFace Inference API

---

## Project Structure

```
NexaServeAI
│
├── chatbot.py
├── requirements.txt
├── docs/
│   └── Onelap Support DOC V1.pdf
└── .gitignore
```

---

## Installation

Clone the repository

```
git clone https://github.com/MadhavAdlakha04/NexaServeAI.git
cd NexaServeAI
```

Install dependencies

```
pip install -r requirements.txt
```

Create a `.env` file and add your API keys.

---

## Usage

Run the chatbot

```
python chatbot.py
```

Ask support-related questions based on the provided documentation.

---

## Example Queries

* How do I reset my Onelap account password?
* What subscription plans are available?
* How can I cancel my membership?

---

## Key Features

Retrieval-Augmented Generation (RAG) pipeline
Enterprise document search using FAISS
Context-aware response generation
Scalable architecture for internal knowledge bases

---

## Future Improvements

Web UI using React
Streaming responses
Hybrid search (BM25 + vector search)
Multi-document support

---

## Author

Madhav Adlakha
