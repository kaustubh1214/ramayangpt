from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import numpy as np
import google.generativeai as genai

# ---------------- FastAPI Setup ----------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ---------------- Gemini Setup ----------------
genai.configure(api_key="AIzaSyCrT399dbKfxUCSUdtdUE-hd9eNNUV7xG8")
gemini = genai.GenerativeModel("models/gemini-2.0-flash")

# ---------------- MongoDB Atlas Setup ----------------
MONGO_URI = "mongodb+srv://kaustubhshinde24:bas7.f.i6iDX8VK@cluster0.1bntd.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client["RamayanaDB"]
collection = db["VersesCollection"]

# ---------------- Fetch Verses ----------------
def fetch_verses():
    return list(collection.find())

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
    best_score = float(sim_scores[best_index])

    # Only return if similarity is high enough
    if best_score < 0.3:
        return None
    return top_verses[best_index]

# ---------------- Load Model + Data ----------------
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

    # Step 1: Ask Gemini to classify the message type
    classify_prompt = f"""
You are a classifier assistant. A user has asked the following question about the Ramayana:

"{message}"

Classify it into one of the following types:
1. General greeting or vague/broad question (e.g. "hi", "tell me about Ramayana", "who is Rama?")
2. Specific question that may relate to a verse (e.g. "What did Rama say to Lakshmana before exile?")

Reply with only the type number (1 or 2).
"""
    try:
        classification = gemini.generate_content(classify_prompt).text.strip()
    except Exception as e:
        return JSONResponse({"reply": f"Gemini classification error: {e}"})

    if classification == "1":
        # Handle general greeting or open question using Gemini only
        try:
            direct_response = gemini.generate_content(
                f"""You are a helpful Ramayana scholar. Answer the user's question in a friendly and informative way:
User: {message}
"""
            )
            reply = direct_response.text.strip()
        except Exception as e:
            reply = f"Gemini error: {e}"
        return JSONResponse({"reply": reply})

    # Step 2: If it's a specific query, try DB + Gemini verification
    top_result = search_bm25_bert(message, verses, bm25, corpus)
    if not top_result:
        return JSONResponse({"reply": "I couldn't find anything in the Ramayana related to your question. Try asking about a character, event, or theme."})

    verse_prompt = f"""
You are a Ramayana expert assistant.

User's Question:
"{message}"

Refer to this related verse for context:

📖 Book: {top_result.get("book", "N/A")}
📘 Chapter: {top_result.get("chapter", "N/A")}
📙 Verse: {top_result.get("verse", "N/A")}
📚 Dictionary: {top_result.get("wordDictionary", "N/A")}
📝 Translation: {top_result.get("translation", "N/A")}

Now answer the user's question based on the verse and its context in Ramayana. Explain clearly and in simple English.
"""
    try:
        final_response = gemini.generate_content(verse_prompt)
        reply = final_response.text.strip()
    except Exception as e:
        reply = f"Gemini error while answering: {e}"

    return JSONResponse({"reply": reply})
