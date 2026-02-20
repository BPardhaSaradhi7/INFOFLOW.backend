from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import fitz
import faiss
import numpy as np
import ollama

app = FastAPI(title="InfoFlow AI API")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Load Embedding Model ----------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- Globals ----------------
index = None
chunks = None

# ---------------- Config ----------------
MODEL_NAME = "phi3"
TOP_K = 2
DISTANCE_THRESHOLD = 1.5

# ---------------- Greeting Responses ----------------
GREETING_RESPONSES = {
    "hi": "Hi 👋 How can I help you today?",
    "hello": "Hello! How can I assist you?",
    "hey": "Hey there! What would you like to know?",
    "good morning": "Good morning ☀️ How can I help you?",
    "good evening": "Good evening! What can I do for you?",
    "good afternoon": "Good afternoon! How can I assist you?",
    "thanks": "You're welcome 😊",
    "thank you": "You're welcome! Happy to help.",
    "how are you": "I'm doing great! How can I help you today?"
}

# ---------------- Helper: Greeting Detection ----------------
def handle_greetings(query: str):
    normalized = query.lower().strip()
    for key in GREETING_RESPONSES:
        if key in normalized:
            return GREETING_RESPONSES[key]
    return None

# ---------------- Helper: Summary Detection ----------------
def is_summary_request(query: str) -> bool:
    summary_keywords = [
        "summary", "summarize", "brief", "overview",
        "give summary", "short summary"
    ]
    query = query.lower()
    return any(word in query for word in summary_keywords)

# ---------------- PDF Extraction ----------------
def extract_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []

    for i in range(len(doc)):
        text = doc[i].get_text().strip()
        if text:
            pages.append({
                "text": text,
                "page": i + 1
            })

    return pages

# ---------------- Chunking ----------------
def chunk_text(pages, chunk_size=500, overlap=50):
    output_chunks = []

    for page in pages:
        words = page["text"].split()
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])

            output_chunks.append({
                "text": chunk,
                "page": page["page"]
            })

            start += chunk_size - overlap

    return output_chunks

# ---------------- Upload Endpoint ----------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global index, chunks

    file_bytes = await file.read()
    pages = extract_pdf(file_bytes)
    chunks = chunk_text(pages)

    texts = [c["text"] for c in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return {
        "message": "PDF processed successfully",
        "total_chunks": len(chunks)
    }

# ---------------- Ask Endpoint ----------------
@app.post("/ask")
async def ask_question(data: dict):
    global index, chunks

    query = data.get("question", "").strip()

    # 1️⃣ Greeting Handling
    greeting_response = handle_greetings(query)
    if greeting_response:
        return {
            "answer": greeting_response,
            "sources": []
        }

    # 2️⃣ No Document Uploaded
    if index is None:
        return {
            "answer": "Please upload a document first so I can assist you.",
            "sources": []
        }

    # 3️⃣ Summary Request Handling
    if is_summary_request(query):
        full_document_text = " ".join([c["text"] for c in chunks[:5]])

        summary_prompt = f"""
You are an AI assistant.

Provide a concise summary in key bullet points based strictly on the document content below.

Document Content:
{full_document_text}
"""

        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": summary_prompt}]
            )
            answer_text = response["message"]["content"]
        except Exception:
            answer_text = "Model error occurred while generating summary."

        return {
            "answer": answer_text,
            "sources": []
        }

    # 4️⃣ Normal Query → RAG Retrieval
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    distances, indices = index.search(query_embedding, min(TOP_K, len(chunks)))
    best_distance = distances[0][0]

    # 5️⃣ Unrelated Query Check
    if best_distance > DISTANCE_THRESHOLD:
        return {
            "answer": (
                "I can only answer questions related to the uploaded document. "
                "Please ask something based on the document content."
            ),
            "sources": []
        }

    # 6️⃣ Build Context
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join([c["text"] for c in retrieved_chunks])

    prompt = f"""
You are an AI assistant answering questions strictly from company documents.

Rules:
- Use ONLY the provided context.
- Do NOT use outside knowledge.
- If answer not found, say:
  "Information not available in the uploaded document."

Context:
{context}

Question:
{query}
"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        answer_text = response["message"]["content"]
    except Exception:
        answer_text = "Model error occurred. Please try again."

    unique_pages = sorted(set(c["page"] for c in retrieved_chunks))

    return {
        "answer": answer_text,
        "sources": unique_pages
    }
