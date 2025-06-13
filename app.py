import os
import json
import re
import time
import base64
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
from dotenv import load_dotenv

# Import enhanced embedding functions
from embedding import (
    enhanced_semantic_search,
    generate_answer,
    embed_with_retry,
    embed_image_with_retry,
    test_connection,
    initialize_system
)

load_dotenv()

# Enhanced Models
class QueryRequest(BaseModel):
    question: str
    source_filter: Optional[str] = None  # "discourse", "markdown", or None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    top_k: Optional[int] = 5

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]
    source_breakdown: Dict[str, int]
    search_time: float
    total_results: int

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced IITM RAG API",
    description="Advanced API for querying IITM course materials, discussions, and images",
    version="2.0.0"
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced async wrappers
async def async_semantic_search(query: str, top_k: int = 5, source_filter=None, image_url=None, image_base64=None):
    """Enhanced async wrapper for semantic search"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, enhanced_semantic_search, query, top_k, source_filter, image_url, image_base64
    )

async def async_generate_answer(query: str, context_texts: List[str]):
    """Enhanced async wrapper for answer generation"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_answer, query, context_texts)

def extract_links_from_results(results):
    """Enhanced link extraction with better URL handling"""
    links = []
    seen = set()
    for result in results:
        url = ""
        text = ""
        
        if result['source_type'] == 'discourse':
            topic_id = result.get('topic_id', '')
            post_number = result.get('root_post_number', '')
            url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}/{post_number}"
            text = result.get('topic_title', 'Discourse Discussion')[:100]
        elif result['source_type'] == 'image':
            url = result.get('image_url', '#')
            text = result.get('title', 'Image')[:100]
        else:  # markdown
            url = result.get('original_url', '#')
            text = result.get('title', 'Course Material')[:100]
        
        if url and url != '#' and url not in seen:
            links.append({"url": url, "text": text})
            seen.add(url)
    return links

def extract_links_from_answer(answer):
    """Extract markdown links from the answer text"""
    pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
    return [{"url": url, "text": text} for text, url in re.findall(pattern, answer)]

def merge_links(links1, links2):
    """Merge two lists of links, avoiding duplicates"""
    seen = set()
    merged = []
    for link in links1 + links2:
        if link['url'] not in seen:
            merged.append(link)
            seen.add(link['url'])
    return merged

@app.post("/query")
async def query_rag_system(request: QueryRequest):
    """Enhanced query endpoint with timing and better error handling"""
    start_time = time.time()
    
    try:
        # Validate input
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Perform semantic search
        results = await async_semantic_search(
            request.question,
            top_k=request.top_k or 5,
            source_filter=request.source_filter,
            image_url=request.image_url,
            image_base64=request.image_base64
        )

        search_time = time.time() - start_time

        if not results:
            response = {
                "answer": "I couldn't find any relevant information in the knowledge base.",
                "links": [],
                "source_breakdown": {"discourse": 0, "markdown": 0, "image": 0},
                "search_time": search_time,
                "total_results": 0
            }
            return JSONResponse(content=jsonable_encoder(response))

        # Generate answer
        context_texts = [res["combined_text"] for res in results]
        answer = await async_generate_answer(request.question, context_texts)

        # Extract and merge links
        links_from_results = extract_links_from_results(results)
        links_from_answer = extract_links_from_answer(answer)
        links = merge_links(links_from_results, links_from_answer)

        # Count sources
        source_breakdown = {"discourse": 0, "markdown": 0, "image": 0}
        for result in results:
            source_type = result.get('source_type', 'discourse')
            if source_type not in source_breakdown:
                source_breakdown[source_type] = 0
            source_breakdown[source_type] += 1

        response = {
            "answer": answer,
            "links": links,
            "source_breakdown": source_breakdown,
            "search_time": round(search_time, 3),
            "total_results": len(results)
        }
        return JSONResponse(content=jsonable_encoder(response))

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Enhanced health check with detailed status"""
    try:
        connection_status = test_connection()
        return {
            "status": "healthy" if connection_status else "degraded",
            "timestamp": time.time(),
            "services": {
                "openai_proxy": "connected" if connection_status else "error",
                "pinecone": "connected" if connection_status else "error"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/stats")
async def get_stats():
    """Get index statistics"""
    try:
        from enhanced_embedding import index
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "index_fullness": stats.index_fullness,
            "dimension": 1536
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Enhanced IITM RAG API v2.0",
        "features": [
            "Text and image query support",
            "Multi-source search (discourse, markdown)",
            "Advanced chunking and retrieval",
            "Enhanced error handling"
        ],
        "docs": "/docs"
    }

# Initialize system on startup
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    print("🚀 Starting Enhanced RAG API...")
    if initialize_system():
        print("✅ System initialized successfully")
    else:
        print("⚠️  System initialization completed with warnings")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
