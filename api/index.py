import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from dotenv import load_dotenv

from embedding import (
    enhanced_semantic_search,
    generate_answer,
    test_connection
)

load_dotenv()

# Request and Response Models
class QueryRequest(BaseModel):
    question: str
    source_filter: Optional[str] = None

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]
    source_breakdown: Dict[str, int]

# Initialize FastAPI app
app = FastAPI(
    title="IITM RAG API",
    description="API for querying IITM course materials and discussions"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Async wrappers
async def async_semantic_search(query: str, top_k: int = 5, source_filter=None):
    return await asyncio.to_thread(enhanced_semantic_search, query, top_k, source_filter)

async def async_generate_answer(query: str, context_texts: List[str]):
    return await asyncio.to_thread(generate_answer, query, context_texts)

def extract_links_from_results(results):
    links = []
    for result in results:
        if result['source_type'] == 'discourse':
            url = f"https://discourse.onlinedegree.iitm.ac.in/t/{result.get('topic_id', '')}/{result.get('root_post_number', '')}"
            text = result.get('topic_title', 'Discourse Discussion')[:100]
        else:
            url = result.get('original_url', '#')
            text = result.get('title', 'Course Material')[:100]

        if url and url != '#':
            links.append({"url": url, "text": text})
    return links

# Main RAG endpoint
@app.post("/query", response_model=QueryResponse)
async def query_rag_system(request: QueryRequest):
    try:
        results = await async_semantic_search(request.question, top_k=5, source_filter=request.source_filter)

        if not results:
            return QueryResponse(
                answer="I couldn't find any relevant information in the knowledge base.",
                links=[],
                source_breakdown={"discourse": 0, "markdown": 0}
            )

        context_texts = [r["combined_text"] for r in results]
        answer = await async_generate_answer(request.question, context_texts)
        links = extract_links_from_results(results)

        source_breakdown = {"discourse": 0, "markdown": 0}
        for r in results:
            src = r.get('source_type', 'discourse')
            if src in source_breakdown:
                source_breakdown[src] += 1

        return QueryResponse(answer=answer, links=links, source_breakdown=source_breakdown)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Health check
@app.get("/health")
async def health_check():
    try:
        connected = test_connection()
        return {
            "status": "healthy" if connected else "degraded",
            "openai_proxy": "connected",
            "pinecone": "connected" if connected else "error"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/")
async def root():
    return {"message": "IITM RAG API is running", "docs": "/docs"}

# For local development

