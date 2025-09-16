import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi import Form
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import traceback

# -------------------
# Load environment
# -------------------
load_dotenv()
CLIENT_SECRETS_FILE = "credentials.json"
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/documents.readonly",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8000/oauth2callback")

# -------------------
# App setup
# -------------------
app = FastAPI()
FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# In-memory stores
USER_CREDENTIALS = {}
VECTOR_STORE = {}  # {user_key: {"index": faiss_index, "texts": [chunks]}}


# -------------------
# Serve frontend
# -------------------
@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


# -------------------
# Google OAuth login
# -------------------
@app.get("/login")
async def login():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, state = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", prompt="consent"
    )
    return RedirectResponse(auth_url)


@app.get("/oauth2callback")
async def oauth2callback(request: Request):
    try:
        full_url = str(request.url)
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        flow.fetch_token(authorization_response=full_url)
        creds = flow.credentials

        USER_CREDENTIALS["default_user"] = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes
        }
        return JSONResponse({"status": "connected", "note": "credentials saved"})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# -------------------
# Helper to get credentials
# -------------------
def get_credentials_for_user(user_key="default_user"):
    info = USER_CREDENTIALS.get(user_key)
    if not info:
        return None
    return Credentials(
        token=info["token"],
        refresh_token=info.get("refresh_token"),
        token_uri=info["token_uri"],
        client_id=info["client_id"],
        client_secret=info["client_secret"],
        scopes=info["scopes"],
    )


# -------------------
# List Google Docs
# -------------------
@app.get("/list_docs")
async def list_docs():
    creds = get_credentials_for_user()
    if not creds:
        return JSONResponse({"error": "not authenticated. visit /login first"}, status_code=401)
    try:
        drive_svc = build("drive", "v3", credentials=creds)
        results = drive_svc.files().list(
            q="mimeType='application/vnd.google-apps.document'",
            fields="files(id, name, modifiedTime, owners)"
        ).execute()
        files = results.get("files", [])
        docs = [{"id": f["id"], "name": f["name"]} for f in files]
        return {"documents": docs}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# -------------------
# Fetch single doc content
# -------------------
@app.get("/fetch_doc/{doc_id}")
async def fetch_doc(doc_id: str):
    creds = get_credentials_for_user()
    if not creds:
        return JSONResponse({"error": "not authenticated"}, status_code=401)

    docs_svc = build("docs", "v1", credentials=creds)
    doc = docs_svc.documents().get(documentId=doc_id).execute()

    def extract_text(gdoc):
        text_chunks = []
        for element in gdoc.get("body", {}).get("content", []):
            if "paragraph" in element:
                for elem in element["paragraph"].get("elements", []):
                    if "textRun" in elem and "content" in elem["textRun"]:
                        text_chunks.append(elem["textRun"]["content"])
        return "".join(text_chunks)

    plain_text = extract_text(doc)
    return {"id": doc.get("documentId"), "title": doc.get("title"), "text": plain_text}


# -------------------
# Ingest docs for RAG
# -------------------
@app.post("/ingest")
async def ingest_docs(doc_ids: list[str]):
    user_key = "default_user"
    creds = get_credentials_for_user(user_key)
    if not creds:
        return JSONResponse({"error": "not authenticated. visit /login"}, status_code=401)

    docs_svc = build("docs", "v1", credentials=creds)
    all_chunks = []

    for doc_id in doc_ids:
        doc = docs_svc.documents().get(documentId=doc_id).execute()
        text_chunks = []
        for element in doc.get("body", {}).get("content", []):
            if "paragraph" in element:
                for elem in element["paragraph"].get("elements", []):
                    if "textRun" in elem and "content" in elem["textRun"]:
                        text_chunks.append(elem["textRun"]["content"])
        plain_text = "".join(text_chunks)
        all_chunks.append(plain_text)

    splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
    chunks = []
    for txt in all_chunks:
        chunks.extend(splitter.split_text(txt))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    VECTOR_STORE[user_key] = {"index": index, "texts": chunks}
    return {"message": f"{len(chunks)} chunks ingested and indexed for RAG."}

# -------------------
# Serve chat page
# -------------------
@app.get("/chat")
async def serve_chat():
    return FileResponse(FRONTEND_DIR / "chat.html")


# -------------------
# Handle chat queries
# -------------------
@app.post("/chat_query")
async def chat_query(query: str = Form(...)):
    user_key = "default_user"
    store = VECTOR_STORE.get(user_key)
    if not store:
        return JSONResponse({"error": "No documents ingested yet. Use /ingest first."}, status_code=400)

    index = store["index"]
    texts = store["texts"]

    # Embed query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query], convert_to_numpy=True)

    # Search FAISS
    D, I = index.search(query_vec, k=3)  # top 3 results
    results = [texts[i] for i in I[0]]

    response_text = "\n---\n".join(results)
    return {"query": query, "response": response_text}