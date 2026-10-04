"""FastAPI backend: runs the multi-agent research graph.

Run (from the project1 folder):
    uvicorn api:app --port 8000
"""
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from research_graph import research_graph, unique_sources   # slow: loads models + vector store once

app = FastAPI(title="Research Assistant API")


class AskRequest(BaseModel):
    question: str


class Evidence(BaseModel):
    sub_question: str
    tool: str | None = None
    text: str


class AskResponse(BaseModel):
    answer: str                  # markdown, with [n] citations and a references list
    sub_questions: list[str]     # planner's + reflector's follow-ups
    evidence: list[Evidence]     # unique evidence the researchers kept
    reflections: int             # how many times the reflector reviewed the answer
    decision: str                # reflector's last decision


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    try:
        result = research_graph.invoke({"questions": [HumanMessage(req.question)]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"The research graph failed: {e}")

    return AskResponse(
        answer=result["answers"][-1].content if result["answers"] else "No answer was produced.",
        sub_questions=result["sub_questions"],
        evidence=unique_sources(result["evidence"]),
        reflections=result["reflections"],
        decision=result["decision"],
    )
