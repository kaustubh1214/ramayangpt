from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.hash import bcrypt
from jose import jwt
import os

app = FastAPI()

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URL = "mongodb+srv://kaustubhshinde24:bas7.f.i6iDX8VK@cluster0.1bntd.mongodb.net/"
client = AsyncIOMotorClient(MONGO_URL)
db = client["ramayanDB"]

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
