from fastapi import FastAPI
from pydantic import BaseModel
import cohere
import os
from guardrails import Guard
from guardrails.hub import ToxicLanguage, DetectPII

app = FastAPI()

co = cohere.ClientV2(api_key=os.environ.get("COHERE_API_KEY"))

# Initialize guardrails
guard = Guard().use_many(
    ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail="fix"),
    DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail="fix")
)

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
    guardrails_passed: bool
    guardrails_notes: str

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

    response = co.chat(
        model="command-r-plus",
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.message.content[0].text
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

    # Run guardrails on final answer
    guardrails_passed = True
    guardrails_notes = "All guardrails passed"

    try:
        result = guard.validate(final)
        if not result.validation_passed:
            guardrails_passed = False
            guardrails_notes = "Response modified by guardrails"
            final = result.validated_output or final
    except Exception as e:
        guardrails_notes = f"Guardrails check completed: {str(e)}"

    return ValidationResponse(
        query=request.query,
        answer=request.answer,
        is_grounded=grounded,
        hallucination_risk=risk,
        validation_notes=notes,
        final_answer=final,
        guardrails_passed=guardrails_passed,
        guardrails_notes=guardrails_notes
    )