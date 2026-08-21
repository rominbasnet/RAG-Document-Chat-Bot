# RAG-Document-Chat-Bot

A modular **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions about documents and receive answers based on the information contained in those documents.

The project uses document ingestion, chunking, embeddings, vector search, and an LLM to retrieve relevant information before generating an answer.

## Features

* 📄 Document ingestion and processing
* ✂️ Text chunking for efficient retrieval
* 🧠 Embedding generation
* 🔎 Semantic/vector-based document retrieval
* 💬 Question answering using an LLM
* 🐍 Python-based backend
* 🐳 Docker Compose support
* 🗂️ Modular RAG architecture

## Architecture

```text
Documents
    │
    ▼
Document Ingestion
    │
    ▼
Text Chunking
    │
    ▼
Embeddings
    │
    ▼
Vector Database
    │
    ▼
User Question
    │
    ▼
Retriever
    │
    ▼
Relevant Document Chunks
    │
    ▼
LLM
    │
    ▼
Generated Answer
```

## Project Structure

```text
RAG-Document-Chat-Bot/
│
├── docs/
│   └── NIST.SP.800-53r5.pdf
│
├── chunking.py
├── db.py
├── embeddings.py
├── ingest.py
├── main.py
├── retrieval.py
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

### Main Components

**`ingest.py`**
Loads and processes documents before they are added to the RAG pipeline.

**`chunking.py`**
Splits large documents into smaller text chunks that can be efficiently retrieved.

**`embeddings.py`**
Converts text chunks into vector representations for semantic search.

**`db.py`**
Handles the vector database/storage layer.

**`retrieval.py`**
Searches the document collection and retrieves the most relevant chunks for a user query.

**`main.py`**
Entry point for running the application.

## Requirements

* Python 3.10+
* Git
* Docker and Docker Compose (if using the provided Docker configuration)
* An LLM provider or locally running LLM, depending on your configuration

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rominbasnet/RAG-Document-Chat-Bot.git
cd RAG-Document-Chat-Bot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

Run the application with:

```bash
python main.py
```

If your project uses Docker:

```bash
docker compose up
```

## RAG Pipeline

The application follows these general steps:

1. **Load documents**
   Documents are loaded from the document directory.

2. **Chunk documents**
   Large documents are divided into smaller sections.

3. **Generate embeddings**
   Each chunk is converted into a numerical vector.

4. **Store embeddings**
   The vectors are stored in a vector database.

5. **Receive a question**
   The user submits a natural-language question.

6. **Retrieve relevant content**
   The system performs semantic search to find relevant document chunks.

7. **Generate an answer**
   The retrieved context is provided to the LLM, which generates the final response.

## Example

A user could ask:

```text
What are the security controls defined in NIST SP 800-53?
```

The system retrieves relevant sections from the document and uses those sections as context when generating the answer.

## Environment Variables

If your selected LLM or embedding provider requires API credentials, store them in a `.env` file rather than committing them to Git.

Example:

```env
API_KEY=your_api_key_here
```

Make sure `.env` is included in `.gitignore`.

## Future Improvements

Potential extensions for this project include:

* [ ] LangGraph-based agentic workflow
* [ ] Conversation memory
* [ ] Human-in-the-loop query clarification
* [ ] Query rewriting
* [ ] Retrieval quality evaluation
* [ ] Support for multiple document formats
* [ ] Streaming responses
* [ ] Web-based chat interface
* [ ] Authentication and user management
* [ ] Persistent conversation history

## Technologies

* **Python**
* **RAG**
* **Embeddings**
* **Vector Database**
* **LLM**
* **Docker**
* **LangChain/LangGraph** *(for future agentic workflow integration)*

## License

This project is intended for educational and development purposes. Add an appropriate open-source license if you plan to distribute the project publicly.
