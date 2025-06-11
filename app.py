from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import asyncio
import time
from datetime import datetime

# OpenAI and Pinecone imports
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize clients
openai_client = OpenAI(
    api_key=AIPROXY_TOKEN,
    base_url="https://aiproxy.sanand.workers.dev/openai/v1"
)

pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index("discourse-embeddings")

# FastAPI app
app = FastAPI(
    title="IITM RAG API",
    description="API for querying IITM course materials and discussions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    source_filter: Optional[str] = None  # "discourse", "markdown", or None
    top_k: Optional[int] = 5

class LinkInfo(BaseModel):
    url: str
    text: str
    source_type: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]
    source_breakdown: Dict[str, int]
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    openai_status: str
    pinecone_status: str
    index_stats: Dict[str, Any]

# Helper functions
async def embed_with_retry(text: str, max_retries: int = 3):
    """Get embedding with retry logic"""
    for attempt in range(max_retries):
        try:
            response = openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
            else:
                raise e

async def enhanced_semantic_search(query: str, top_k: int = 5, source_filter=None) -> List[Dict[str, Any]]:
    """Enhanced search that can filter by source type"""
    try:
        # Get query embedding
        query_embedding = await embed_with_retry(query)
        
        # Build filter if specified
        filter_dict = {}
        if source_filter:
            filter_dict["source_type"] = {"$eq": source_filter}
        
        # Search Pinecone
        search_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        results = []
        for match in search_response.matches:
            result = {
                "score": match.score,
                "source_type": match.metadata.get("source_type", "discourse"),
                "combined_text": match.metadata["combined_text"]
            }
            
            # Add source-specific metadata
            if match.metadata.get("source_type") == "markdown":
                result.update({
                    "title": match.metadata.get("title", ""),
                    "original_url": match.metadata.get("original_url", ""),
                    "chunk_index": match.metadata.get("chunk_index", 0)
                })
            else:  # discourse
                result.update({
                    "topic_id": match.metadata.get("topic_id", ""),
                    "topic_title": match.metadata.get("topic_title", ""),
                    "root_post_number": match.metadata.get("root_post_number", "")
                })
            
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Error in enhanced search: {e}")
        return []

async def generate_answer(query: str, context_texts: List[str]) -> str:
    """Generate answer using OpenAI"""
    try:
        context = "\n\n---\n\n".join(context_texts)
        
        # Limit context length
        if len(context) > 4000:
            context = context[:4000] + "..."

        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant that answers questions based on IITM course materials and forum discussions. Provide clear, accurate answers based on the given context. If the information comes from different sources (forum discussions vs course materials), mention this in your response."
            },
            {
                "role": "user", 
                "content": f"Based on these excerpts:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
            }
        ]

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Updated model name
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer due to an error."

def extract_links_from_results(results: List[Dict[str, Any]]) -> List[LinkInfo]:
    """Extract links from search results"""
    links = []
    seen_urls = set()
    
    for result in results:
        if result['source_type'] == 'discourse':
            # Create discourse URL
            topic_id = result.get('topic_id', '')
            post_number = result.get('root_post_number', '')
            if topic_id and post_number:
                url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}/{post_number}"
                text = result.get('topic_title', 'Discourse Discussion')[:100]
                source_type = "discourse"
        else:  # markdown
            url = result.get('original_url', '')
            text = result.get('title', 'Course Material')[:100]
            source_type = "markdown"
        
        if url and url not in seen_urls and url != '#':
            links.append(LinkInfo(url=url, text=text, source_type=source_type))
            seen_urls.add(url)
    
    return links

# API Routes
@app.get("/", response_model=dict)
async def root():
    return {
        "message": "IITM RAG API is running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        # Test OpenAI connection
        openai_status = "connected"
        try:
            await embed_with_retry("test")
        except:
            openai_status = "error"
        
        # Test Pinecone connection
        pinecone_status = "connected"
        index_stats = {}
        try:
            stats = index.describe_index_stats()
            index_stats = {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension
            }
        except:
            pinecone_status = "error"
        
        status = "healthy" if openai_status == "connected" and pinecone_status == "connected" else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            openai_status=openai_status,
            pinecone_status=pinecone_status,
            index_stats=index_stats
        )
        
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            openai_status="error",
            pinecone_status="error",
            index_stats={"error": str(e)}
        )

@app.post("/query", response_model=QueryResponse)
async def query_rag_system(request: QueryRequest):
    start_time = time.time()
    
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Perform semantic search
        results = await enhanced_semantic_search(
            request.question, 
            top_k=request.top_k, 
            source_filter=request.source_filter
        )
        
        if not results:
            processing_time = time.time() - start_time
            return QueryResponse(
                answer="I couldn't find any relevant information in the knowledge base for your question.",
                links=[],
                source_breakdown={"discourse": 0, "markdown": 0},
                processing_time=processing_time
            )
        
        # Generate answer
        context_texts = [res["combined_text"] for res in results]
        answer = await generate_answer(request.question, context_texts)
        
        # Extract links
        links = extract_links_from_results(results)
        
        # Count sources
        source_breakdown = {"discourse": 0, "markdown": 0}
        for result in results:
            source_type = result.get('source_type', 'discourse')
            if source_type in source_breakdown:
                source_breakdown[source_type] += 1
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=answer,
            links=links,
            source_breakdown=source_breakdown,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness,
            "namespaces": stats.namespaces
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
