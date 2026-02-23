# Multi-Agent RAG Pipeline with services containarized on docker and orchestrated on Kubernetes

A production-grade multi-agent Retrieval Augmented Generation (RAG) 
pipeline where each agent runs as an independent microservice 
orchestrated on Kubernetes.

## Architecture
User Query
    ↓
Orchestrator (Port 8000)
    ↓
Retrieval Agent (FAISS vector search) → Port 8001
    ↓
Reasoning Agent (Cohere command-r-plus) → Port 8002
    ↓
Validation Agent (Hallucination detection) → Port 8003
    ↓
Final grounded response


## Tech Stack
- FastAPI — REST API for each agent
- FAISS — Local vector similarity search
- Cohere — LLM for reasoning and validation
- Docker— Each agent containerized independently
- Kubernetes — Orchestration, service discovery, health monitoring

## Agents
- Retrieval Agent — Embeds query, searches FAISS index, returns top-k chunks
- easoning Agent — Takes retrieved chunks, generates grounded answer via Cohere
- Validation Agent — Checks answer for hallucinations, scores confidence
- Orchestrator — Coordinates all three agents, returns final response

## Features
- Liveness and readiness probes on every agent
- Kubernetes service discovery between agents
- Hallucination detection and grounding validation
- Horizontally scalable — each agent scales independently
- Secret management via Kubernetes secrets

## Setup

### Prerequisites
- Docker Desktop
- Minikube
- Cohere API key

### Run locally

1. Start minikube
```bash
minikube start --driver=docker
minikube docker-env | Invoke-Expression
```

2. Build all images
```bash
docker build -t retrieval-agent:latest ./retrieval-agent
docker build -t reasoning-agent:latest ./reasoning-agent
docker build -t validation-agent:latest ./validation-agent
docker build -t orchestrator:latest ./orchestrator
```

3. Create secret
```bash
cp k8s/secret.example.yaml k8s/secret.yaml

```

4. Deploy to Kubernetes
```bash
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/retrieval-deployment.yaml
kubectl apply -f k8s/reasoning-deployment.yaml
kubectl apply -f k8s/validation-deployment.yaml
kubectl apply -f k8s/orchestrator-deployment.yaml
```

5. Access the API
```bash
minikube service orchestrator --url
```

### Example Query
```bash
curl -X POST http://localhost:PORT/query \
  -H "Content-Type: application/json" \
  -d '{"query": "how does KLA detect wafer defects", "top_k": 3}'
```

### Example Response
```json
{
  "query": "how does KLA detect wafer defects",
  "retrieved_chunks": [...],
  "raw_answer": "...",
  "final_answer": "...",
  "is_grounded": true,
  "hallucination_risk": "low",
  "validation_notes": "...",
  "confidence": 0.95
}
```
