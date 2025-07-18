import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
from dotenv import load_dotenv

# Import your existing RAG functions
from embedding import (
    enhanced_semantic_search,
    generate_answer,
    embed_with_retry,
    test_connection
)

load_dotenv()

# Models


class QueryRequest(BaseModel):
    question: str
    # "discourse", "markdown", or None for both
    source_filter: Optional[str] = None


class LinkInfo(BaseModel):
    url: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]
    source_breakdown: Dict[str, int]


# Initialize FastAPI app
app = FastAPI(title="IITM RAG API",
              description="API for querying IITM course materials and discussions")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Convert your sync functions to async


async def async_semantic_search(query: str, top_k: int = 5, source_filter=None):
    """Async wrapper for your semantic search"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, enhanced_semantic_search, query, top_k, source_filter)


async def async_generate_answer(query: str, context_texts: List[str]):
    """Async wrapper for your answer generation"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_answer, query, context_texts)


def extract_links_from_results(results):
    """Extract links from search results"""
    links = []
    for result in results:
        if result['source_type'] == 'discourse':
            # Create discourse URL
            topic_id = result.get('topic_id', '')
            post_number = result.get('root_post_number', '')
            url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}/{post_number}"
            text = result.get('topic_title', 'Discourse Discussion')[:100]
        else:  # markdown
            url = result.get('original_url', '#')
            text = result.get('title', 'Course Material')[:100]

        if url and url != '#':
            links.append({"url": url, "text": text})

    return links


@app.post("/query", response_model=QueryResponse)
async def query_rag_system(request: QueryRequest):
    try:
        # Perform semantic search
        results = await async_semantic_search(
            request.question,
            top_k=5,
            source_filter=request.source_filter
        )

        if not results:
            return QueryResponse(
                answer="I couldn't find any relevant information in the knowledge base.",
                links=[],
                source_breakdown={"discourse": 0, "markdown": 0}
            )

        # Generate answer
        context_texts = [res["combined_text"] for res in results]
        answer = await async_generate_answer(request.question, context_texts)

        # Extract links
        links = extract_links_from_results(results)

        # Count sources
        source_breakdown = {"discourse": 0, "markdown": 0}
        for result in results:
            source_type = result.get('source_type', 'discourse')
            source_breakdown[source_type] += 1

        return QueryResponse(
            answer=answer,
            links=links,
            source_breakdown=source_breakdown
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/health")
async def health_check():
    try:
        # Test your connections
        connection_status = test_connection()
        return {
            "status": "healthy" if connection_status else "degraded",
            "openai_proxy": "connected",
            "pinecone": "connected" if connection_status else "error"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/")
async def root():
    return {"message": "IITM RAG API is running", "docs": "/docs"}

# For Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
