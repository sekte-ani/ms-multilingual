from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("intfloat/multilingual-e5-base")

class EmbedRequest(BaseModel):
    keywords: str

@app.post("/embed")
def embed(payload: EmbedRequest):
    text = f"passage: {payload.keywords}"
    vector = model.encode(text, normalize_embeddings=True)
    return { "vector": vector.tolist() }

@app.get("/health")
def health():
    return { "status": "ok" }



