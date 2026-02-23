from fastapi import FastAPI
from pydantic import BaseModel
import cohere
import os

app = FastAPI()

client = cohere.Client(api_key=os.environ.get("COHERE_API_KEY"))

class ValidationRequest(BaseModel):
    query: str
    answer: str
    retrieved_chunks: list[str]

class ValidationResponse(BaseModel):
    query: str
    answer: str
    is_grounded: bool
    hallucination_risk: str
    validation_notes: str
    final_answer: str

@app.get("/health")
def health():
    return {"status": "healthy", "agent": "validation"}

@app.post("/validate", response_model=ValidationResponse)
def validate(request: ValidationRequest):
    context = "\n".join([
        f"- {chunk}" 
        for chunk in request.retrieved_chunks
    ])

    prompt = f"""You are a hallucination detection system. 
Your job is to check if an AI answer is grounded in the provided context.

Context:
{context}

Question: {request.query}

Answer to validate: {request.answer}

Respond in this exact format:
GROUNDED: yes or no
RISK: low, medium, or high
NOTES: one sentence explanation
FINAL: the corrected or confirmed answer"""

    response = client.chat(
        model="command-r-plus-08-2024",
        message=prompt,
        temperature=0.1
    )

    text = response.text
    lines = text.strip().split("\n")

    grounded = True
    risk = "low"
    notes = "Answer appears grounded in context."
    final = request.answer

    for line in lines:
        if line.startswith("GROUNDED:"):
            grounded = "yes" in line.lower()
        elif line.startswith("RISK:"):
            risk = line.replace("RISK:", "").strip()
        elif line.startswith("NOTES:"):
            notes = line.replace("NOTES:", "").strip()
        elif line.startswith("FINAL:"):
            final = line.replace("FINAL:", "").strip()

    return ValidationResponse(
        query=request.query,
        answer=request.answer,
        is_grounded=grounded,
        hallucination_risk=risk,
        validation_notes=notes,
        final_answer=final
    )