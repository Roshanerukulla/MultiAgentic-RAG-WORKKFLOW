from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

# Service URLs — these will be Kubernetes service names
RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL", "http://retrieval-agent:8001")
REASONING_URL = os.environ.get("REASONING_URL", "http://reasoning-agent:8002")
VALIDATION_URL = os.environ.get("VALIDATION_URL", "http://validation-agent:8003")

TIMEOUT = 30.0

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class FinalResponse(BaseModel):
    query: str
    retrieved_chunks: list[str]
    raw_answer: str
    final_answer: str
    is_grounded: bool
    hallucination_risk: str
    validation_notes: str
    confidence: float
    guardrails_passed: bool
    guardrails_notes: str

@app.get("/health")
def health():
    return {"status": "healthy", "agent": "orchestrator"}

@app.post("/query", response_model=FinalResponse)
async def query(request: QueryRequest):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:

        # Step 1 — Retrieval Agent
        try:
            retrieval_response = await client.post(
                f"{RETRIEVAL_URL}/retrieve",
                json={
                    "query": request.query,
                    "top_k": request.top_k
                }
            )
            retrieval_data = retrieval_response.json()
            chunks = retrieval_data["retrieved_chunks"]
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Retrieval agent failed: {str(e)}"
            )

        # Step 2 — Reasoning Agent
        try:
            reasoning_response = await client.post(
                f"{REASONING_URL}/reason",
                json={
                    "query": request.query,
                    "retrieved_chunks": chunks
                }
            )
            reasoning_data = reasoning_response.json()
            raw_answer = reasoning_data["answer"]
            confidence = reasoning_data["confidence"]
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Reasoning agent failed: {str(e)}"
            )

        # Step 3 — Validation Agent
        try:
            validation_response = await client.post(
                f"{VALIDATION_URL}/validate",
                json={
                    "query": request.query,
                    "answer": raw_answer,
                    "retrieved_chunks": chunks
                }
            )
            validation_data = validation_response.json()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Validation agent failed: {str(e)}"
            )

        return FinalResponse(
    query=request.query,
    retrieved_chunks=chunks,
    raw_answer=raw_answer,
    final_answer=validation_data["final_answer"],
    is_grounded=validation_data["is_grounded"],
    hallucination_risk=validation_data["hallucination_risk"],
    validation_notes=validation_data["validation_notes"],
    confidence=confidence,
    guardrails_passed=validation_data["guardrails_passed"],
    guardrails_notes=validation_data["guardrails_notes"]
)