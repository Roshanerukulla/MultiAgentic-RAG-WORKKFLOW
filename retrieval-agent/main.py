from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app= FastAPI()


#loadembedding agent

model = SentenceTransformer("all-MiniLM-L6-v2")


#SampleKnoeedge corpus

documents=[
    "Semiconductor wafer inspection detects defects at nanometer scale using optical and electron beam systems.",
    "KLA systems provide metrology and inspection for every layer of chip manufacturing.",
    "Defect classification uses machine learning models trained on historical wafer scan data.",
    "RAG systems retrieve relevant context before generating answers to improve accuracy.",
    "Docker containers package application code and dependencies into portable units.",
    "Kubernetes orchestrates containers across clusters managing scaling and availability.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "MLOps pipelines automate training validation deployment and monitoring of ML models.",
    "Retrieval augmented generation grounds LLM responses in retrieved factual context.",
    "Production ML systems require drift detection retraining triggers and observability.",

]

#FAISS

embeddings= model.encode(documents)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class RetrievalResponse(BaseModel):
    query: str
    retrieved_chunks: list[str]
    scores: list[float]

@app.get("/health")
def health():
    return {"status": "healthy", "agent": "retrieval"}

@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve(request: QueryRequest):
    query_embedding = model.encode([request.query])
    distances, indices = index.search(
        np.array(query_embedding), 
        request.top_k
    )
    
    retrieved = [documents[i] for i in indices[0]]
    scores = [float(1 / (1 + d)) for d in distances[0]]
    
    return RetrievalResponse(
        query=request.query,
        retrieved_chunks=retrieved,
        scores=scores
    )