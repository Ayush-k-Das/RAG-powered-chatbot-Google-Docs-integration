RAG Chatbot with Google Docs Integration
Overview

This project is a Retrieval-Augmented Generation (RAG) Chatbot that connects to Google Docs, ingests content, and allows you to query your documents via a web-based interface. It uses FastAPI for the backend and a simple HTML/CSS frontend.

The chatbot leverages embedding-based retrieval using SentenceTransformer and FAISS for semantic search.

Features

Google OAuth2 login to access user Google Docs.

List and fetch Google Docs content.

Ingest selected documents and create embeddings.

Query documents via chatbot interface.

Frontend served directly via FastAPI.

Project Structure
rag-chatbot/
│
├─ main.py                # FastAPI backend code
├─ frontend/
│   └─ index.html         # Chat UI
├─ .env                   # Environment variables (ignored in GitHub)
├─ .gitignore             # Ignore virtualenv, credentials, .env, etc.
├─ requirements.txt       # Python dependencies
└─ README.md

Prerequisites

Python 3.11+

Google Cloud Project with OAuth2 credentials

credentials.json file for Google OAuth2 (keep this local, do not push to GitHub)

Basic Python packages:

pip install -r requirements.txt

Setup

Clone the repository

git clone <your-repo-url>
cd rag-chatbot


Create virtual environment

python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate # macOS/Linux


Install dependencies

pip install -r requirements.txt


Add environment variables

Create a .env file:

GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret
REDIRECT_URI=http://localhost:8000/oauth2callback


Place credentials.json locally (ignored by GitHub).

Running the Project
uvicorn main:app --reload --port 8000


Visit http://127.0.0.1:8000
 to open the chatbot frontend.

Use /login to authenticate with Google and access your Docs.

Usage

Click Login with Google.

Select Google Docs to ingest.

Start querying your documents via the chat interface.

Security Notes

Do not push credentials.json or .env to GitHub.

Use .env.example for collaborators to know required variables:

GOOGLE_CLIENT_ID=your_client_id_here
GOOGLE_CLIENT_SECRET=your_client_secret_here
REDIRECT_URI=http://localhost:8000/oauth2callback

Dependencies

FastAPI

Uvicorn

google-auth-oauthlib

google-api-python-client

python-dotenv

sentence-transformers

faiss-cpu

numpy

Future Improvements

Persistent database storage for user credentials and vectors.

Advanced UI with React or Vue.js.

Enhanced error handling and logging.

Deployment on cloud platforms like Heroku, AWS, or Google Cloud.
