"""FastAPI service.

Run:  uvicorn app.api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag import Answer, answer_question

app = FastAPI(title="UK Filings Analyst", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    company: str | None = None
    year: int | None = Field(default=None, ge=2000, le=2100)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    return answer_question(req.question, req.company, req.year, k=get_settings().top_k)
