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

# -------------------- DB Connection --------------------
def get_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='root',
        database='ramayana_db'
    )

# -------------------- Fetch Data --------------------
def fetch_shlokas():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM ramayana_shlokas")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# -------------------- Prepare Corpus --------------------
def combine_fields(entry):
    parts = [
        entry.get("shloka_text", ""),
        entry.get("explanation", ""),
        entry.get("comments", "")
    ]
    return " ".join([p for p in parts if p])

def prepare_bm25_corpus(shlokas):
    corpus = [combine_fields(s) for s in shlokas]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus), corpus

# -------------------- Search with BM25 + BERT --------------------
def search_bm25_bert(query, shlokas, bm25, corpus, top_k=5):
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(bm25_scores)[-top_k:]
    top_shlokas = [shlokas[i] for i in top_indices]
    top_docs = [corpus[i] for i in top_indices]

    query_embedding = bert_model.encode(query, convert_to_tensor=True)
    doc_embeddings = bert_model.encode(top_docs, convert_to_tensor=True)
    sim_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    best_index = int(np.argmax(sim_scores))
    return top_shlokas[best_index]

# -------------------- Load Models and Data --------------------
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
shlokas = fetch_shlokas()
bm25, corpus = prepare_bm25_corpus(shlokas)

# -------------------- Gemini Setup --------------------
genai.configure(api_key="AIzaSyCrT399dbKfxUCSUdtdUE-hd9eNNUV7xG8")  # Replace with your Gemini API key
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

# -------------------- Routes --------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...)):
    if not message:
        return JSONResponse({"reply": "Please enter a valid question."})

    top_result = search_bm25_bert(message, shlokas, bm25, corpus)

    prompt = f"""
You are a knowledgeable Ramayana assistant. A user asked: "{message}"

Refer to the following:

Shloka Text (Sanskrit): {top_result.get("shloka_text", "N/A")}

Explanation: {top_result.get("explanation", "N/A")}

Comments: {top_result.get("comments", "N/A")}

Based on this, provide a clear and contextual answer.
"""

    try:
        response = gemini_model.generate_content(prompt)
        reply = response.text.strip()
    except Exception as e:
        reply = f"Gemini API error: {e}"

    return JSONResponse({"reply": reply})
