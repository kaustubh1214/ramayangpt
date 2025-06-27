from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import mysql.connector
import numpy as np
import google.generativeai as genai

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ---------------- Gemini Setup ----------------
genai.configure(api_key="AIzaSyCrT399dbKfxUCSUdtdUE-hd9eNNUV7xG8")
gemini = genai.GenerativeModel("models/gemini-2.0-flash")

# ---------------- DB Connection ----------------
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="valmiki_ramayana"
    )

# ---------------- Fetch Verses ----------------
def fetch_verses():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM verses")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# ---------------- Corpus Preparation ----------------
def combine_fields(entry):
    return " ".join([
        entry.get("wordDictionary", ""),
        entry.get("translation", "")
    ])

def prepare_bm25(verses):
    corpus = [combine_fields(v) for v in verses]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus), corpus

# ---------------- Search Logic ----------------
def search_bm25_bert(query, verses, bm25, corpus, top_k=5):
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(bm25_scores)[-top_k:]
    top_docs = [corpus[i] for i in top_indices]
    top_verses = [verses[i] for i in top_indices]

    query_embedding = bert_model.encode(query, convert_to_tensor=True)
    doc_embeddings = bert_model.encode(top_docs, convert_to_tensor=True)
    sim_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

    best_index = int(np.argmax(sim_scores))
    return top_verses[best_index]

# ---------------- Load Models + Data ----------------
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
verses = fetch_verses()
bm25, corpus = prepare_bm25(verses)

# ---------------- Routes ----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...)):
    if not message:
        return JSONResponse({"reply": "Please enter a valid question."})

    top_result = search_bm25_bert(message, verses, bm25, corpus)

    prompt = f"""
You are a helpful Ramayana scholar.

User's Question:
"{message}"

Refer to this related verse if relevant:

📖 Book: {top_result.get("book", "N/A")}
📘 Chapter: {top_result.get("chapter", "N/A")}
📙 Verse: {top_result.get("verse", "N/A")}
📚 Dictionary: {top_result.get("wordDictionary", "N/A")}
📝 Translation: {top_result.get("translation", "N/A")}

Based on Ramayana scriptures, explain the answer in proper context in English, in simple yet detailed language.
"""

    try:
        response = gemini.generate_content(prompt)
        reply = response.text.strip()
    except Exception as e:
        reply = f"Gemini API error: {e}"

    return JSONResponse({"reply": reply})
