import os
from fastapi import FastAPI
from pydantic import BaseModel
from .graph import app_graph


app = FastAPI(title="Local Wren-like API")


class ChatRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(body: ChatRequest):
    result = app_graph.invoke({"question": body.query})
    return {
    "answer": result.get("answer", ""),
    "contexts": result.get("contexts", []),
    }