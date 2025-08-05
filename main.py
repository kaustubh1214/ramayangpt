# main.py
from fastapi import FastAPI, HTTPException, Depends, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.hash import bcrypt
from jose import jwt
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import numpy as np
import google.generativeai as genai

# ─────────────────────────────────────────────
# Root app: user auth + simple chat‐history
# ─────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URL = "mongodb+srv://kaustubhshinde24:bas7.f.i6iDX8VK@cluster0.1bntd.mongodb.net/"
motor_client = AsyncIOMotorClient(MONGO_URL)
db = motor_client["ramayanDB"]

SECRET = "supersecret"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    name: str
    email: str
    password: str

@app.post("/signup")
async def signup(user: User):
    existing = await db.users.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = bcrypt.hash(user.password)
    await db.users.insert_one({
        "name": user.name,
        "email": user.email,
        "password": hashed
    })
    return {"message": "Signup successful"}

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await db.users.find_one({"email": form_data.username})
    if not user or not bcrypt.verify(form_data.password, user["password"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = jwt.encode({"sub": user["email"]}, SECRET, algorithm="HS256")
    return {"access_token": token, "token_type": "bearer"}

@app.get("/me")
async def me(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
        email = payload.get("sub")
        return {"email": email}
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
async def chat(input: ChatInput, token: str = Depends(oauth2_scheme)):
    user = await me(token)
    chat_log = {"email": user["email"], "message": input.message}
    await db.chats.insert_one(chat_log)
    # Mock LLM response
    return {"reply": f"Answer to: {input.message}"}

@app.get("/history")
async def history(token: str = Depends(oauth2_scheme)):
    user = await me(token)
    chats = await db.chats.find({"email": user["email"]}).to_list(100)
    return chats

# ─────────────────────────────────────────────
# Sub-app: Ramayan‐specific chatbot logic
# ─────────────────────────────────────────────
chatbot_app = FastAPI()
app.mount("/ramayan", chatbot_app)

templates = Jinja2Templates(directory="templates")

# Gemini setup
genai.configure(api_key="AIzaSyCrT399dbKfxUCSUdtdUE-hd9eNNUV7xG8")
gemini = genai.GenerativeModel("models/gemini-2.0-flash")

# Sync Mongo for verses
mongo_sync = MongoClient(MONGO_URL)
verses_db = mongo_sync["RamayanaDB"]
collection = verses_db["VersesCollection"]

def fetch_verses():
    return list(collection.find())

def combine_fields(entry):
    return " ".join([ entry.get("wordDictionary",""), entry.get("translation","") ])

def prepare_bm25(verses):
    corpus = [combine_fields(v) for v in verses]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus), corpus

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
    if best_score < 0.3:
        return None
    return top_verses[best_index]

bert_model = SentenceTransformer('all-MiniLM-L6-v2')
verses = fetch_verses()
bm25, corpus = prepare_bm25(verses)

@chatbot_app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@chatbot_app.post("/chat")
async def ramayan_chat(message: str = Form(...)):
    if not message:
        return JSONResponse({"reply": "Please enter a valid question."})

    # 1) classify
    classify_prompt = f"""
You are a classifier assistant. A user has asked the following question about the Ramayana:

"{message}"

Classify it into one of the following types:
1. General greeting or vague/broad question
2. Specific question that may relate to a verse

Reply with only the type number (1 or 2).
"""
    try:
        cls = gemini.generate_content(classify_prompt).text.strip()
    except Exception as e:
        return JSONResponse({"reply": f"Gemini classification error: {e}"})

    if cls == "1":
        try:
            direct = gemini.generate_content(
                f"You are a helpful Ramayana scholar. Answer:\nUser: {message}"
            )
            return JSONResponse({"reply": direct.text.strip()})
        except Exception as e:
            return JSONResponse({"reply": f"Gemini error: {e}"})

    # 2) specific
    top = search_bm25_bert(message, verses, bm25, corpus)
    if not top:
        return JSONResponse({"reply": "I couldn't find anything in the Ramayana related to your question."})

    verse_prompt = f"""
You are a Ramayana expert assistant.

User's Question:
"{message}"

Refer to this related verse:

Book: {top.get("book","N/A")}
Chapter: {top.get("chapter","N/A")}
Verse: {top.get("verse","N/A")}
Dictionary: {top.get("wordDictionary","N/A")}
Translation: {top.get("translation","N/A")}

Now answer based on the verse and context.
"""
    try:
        final = gemini.generate_content(verse_prompt)
        return JSONResponse({"reply": final.text.strip()})
    except Exception as e:
        return JSONResponse({"reply": f"Gemini error while answering: {e}"})
