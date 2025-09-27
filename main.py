import os, json, math, uuid, asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncpg

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
RENTCAST_API_KEY = os.getenv("RENTCAST_API_KEY")
CORS = [o.strip() for o in os.getenv("CORS_ORIGINS","").split(",") if o.strip()]

app = FastAPI(title="superior-ai")
app.add_middleware(CORSMiddleware, allow_origins=CORS or ["*"], allow_methods=["*"], allow_headers=["*"])

PG_DSN = os.getenv("PG_DSN")  # e.g. postgres://user:pass@host:5432/db

# ---------- Models
class AskPayload(BaseModel):
    query: str
    address: Optional[str] = None
    k: int = 8

class AiAnswer(BaseModel):
    answer: str
    references: List[str] = Field(default_factory=list)
    data: dict = Field(default_factory=dict)

# ---------- DB / Vector helpers
async def pg():
    return await asyncpg.connect(PG_DSN)

EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4.1-mini"  # adjust as needed

async def embed(texts: List[str]) -> List[List[float]]:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"input": texts, "model": EMBED_MODEL},
        )
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]

async def chat(messages: List[dict]) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": CHAT_MODEL,
                "messages": messages,
                "temperature": 0.2,
            },
        )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ---------- External data fetchers (stubs to wire real calls)
async def get_rentcast(address: str) -> dict:
    # You already use RentCast—swap with your real endpoint + lat/lng
    return {"rentcast": {"address": address, "sample": True}}

async def get_crime_nearby(address: str) -> dict:
    # Replace with your scraper or vendor
    return {"crime": {"address": address, "incidents_12mo": 42}}

# ---------- Retrieval (pgvector)
# SQL table (run once):
# CREATE TABLE IF NOT EXISTS documents(
#   id uuid PRIMARY KEY,
#   address text,
#   title text,
#   content text,
#   source text,
#   embedding vector(3072)
# );
# CREATE INDEX IF NOT EXISTS idx_docs_embedding ON documents USING ivfflat (embedding vector_cosine_ops);

async def retrieve_similar(q_embed: List[float], k: int):
    conn = await pg()
    rows = await conn.fetch(
        "SELECT id, title, content, source FROM documents ORDER BY embedding <=> $1 LIMIT $2",
        q_embed, k
    )
    await conn.close()
    return [dict(r) for r in rows]

# ---------- Routes
@app.get("/")
def root(): return {"message": "Superior AI online"}

@app.post("/ai/ask", response_model=AiAnswer)
async def ai_ask(payload: AskPayload):
    if not OPENAI_API_KEY: raise HTTPException(500, "Missing OPENAI_API_KEY")
    # 1) Embed query
    [q_vec] = await embed([payload.query])
    # 2) Retrieve context
    docs = await retrieve_similar(q_vec, payload.k)
    ctx = "\n\n".join([f"[{d['title']}] {d['content']}" for d in docs])
    # 3) Fetch live data (non-blocking)
    address = payload.address or ""
    rentcast_task = asyncio.create_task(get_rentcast(address)) if address else None
    crime_task = asyncio.create_task(get_crime_nearby(address)) if address else None

    # 4) Compose prompt
    sys = (
      "You are Superior AI, a cautious property analyst. "
      "Answer ONLY from provided context and data; if unsure, say what’s missing. "
      "Always include short bullet references to sources."
    )
    user = f"Question: {payload.query}\n\nContext:\n{ctx[:12000]}"
    if address: user += f"\n\nFocus address: {address}"

    # 5) Call LLM
    text = await chat([{"role":"system","content":sys},{"role":"user","content":user}])

    # 6) Gather live data
    live = {}
    if rentcast_task: live.update(await rentcast_task)
    if crime_task: live.update(await crime_task)

    refs = [d["source"] for d in docs]
    return AiAnswer(answer=text, references=list(dict.fromkeys(refs)), data=live)
