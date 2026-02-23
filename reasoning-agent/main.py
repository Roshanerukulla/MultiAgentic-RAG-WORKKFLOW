from fastapi import FastAPI
from pydantic import BaseModel
import cohere
import os

app = FastAPI()

client = cohere.Client(api_key=os.environ.get("COHERE_API_KEY"))

class ReasoningRequest(BaseModel):
    query: str
    retrieved_chunks: list[str]

class ReasoningResponse(BaseModel):
    query: str
    answer: str
    confidence: float

@app.get("/health")
def health():
    return {"status": "healthy", "agent": "reasoning"}

@app.post("/reason", response_model=ReasoningResponse)
def reason(request: ReasoningRequest):
    context = "\n".join([
        f"- {chunk}" 
        for chunk in request.retrieved_chunks
    ])
    
    prompt = f"""You are an AI assistant answering questions based only on provided context.

Context:
{context}

Question: {request.query}

Answer based strictly on the context above. If the answer isn't in the context, say so clearly."""

    response = client.chat(
        model="command-r-plus-08-2024",
        message=prompt,
        temperature=0.2
    )
    
    answer = response.text
    
    # Simple confidence based on answer length and context match
    confidence = min(0.95, len(answer) / 500)
    
    return ReasoningResponse(
        query=request.query,
        answer=answer,
        confidence=round(confidence, 2)
    )