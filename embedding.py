import os
import time
import base64
import requests
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client with IITM proxy
openai_client = OpenAI(
    api_key=os.getenv("AIPROXY_TOKEN"),
    base_url="https://aiproxy.sanand.workers.dev/openai/v1"
)

# Initialize Pinecone client
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "discourse-embeddings"

# Enhanced index creation with better error handling
try:
    existing_indexes = pinecone.list_indexes().names()
    if index_name not in existing_indexes:
        pinecone.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Successfully created index: {index_name}")
        time.sleep(10)  # Wait for index to be ready
    else:
        print(f"Index {index_name} already exists")
except Exception as e:
    print(f"Error with index operations: {e}")

index = pinecone.Index(index_name)

# ==================== CORE EMBEDDING FUNCTIONS ====================


def embed_with_retry(text: str, max_retries: int = 3):
    """Enhanced text embedding with better error handling"""
    for attempt in range(max_retries):
        try:
            # Clean and validate input
            text = str(text).replace("\n", " ").strip()
            if not text:
                raise ValueError("Empty text input")

            # Truncate if too long
            if len(text) > 8000:
                text = text[:8000] + "..."

            response = openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 60)
                print(
                    f"API error on attempt {attempt + 1}, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                raise e


def embed_image_with_retry(image_url=None, image_base64=None, max_retries=3):
    """Enhanced image embedding with vision API fallback"""
    if not image_url and not image_base64:
        raise ValueError("No image input provided.")

    for attempt in range(max_retries):
        try:
            # Download image if URL provided
            if image_url:
                try:
                    response = requests.get(image_url, timeout=30)
                    response.raise_for_status()
                    image_data = base64.b64encode(
                        response.content).decode('utf-8')
                except requests.RequestException as e:
                    raise ValueError(f"Failed to download image from URL: {e}")
            else:
                image_data = image_base64

            # Try vision API to describe image, then embed description
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image in detail for search and retrieval purposes. Include key visual elements, text content, diagrams, and context."},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"}}
                            ]
                        }
                    ],
                    max_tokens=500
                )
                description = response.choices[0].message.content
                return embed_with_retry(f"Image content: {description}")
            except Exception as vision_error:
                print(f"Vision API failed, using fallback: {vision_error}")
                fallback_text = f"Image from URL: {image_url}" if image_url else "Base64 encoded image"
                return embed_with_retry(fallback_text)

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 60)
                print(
                    f"Image embedding error on attempt {attempt + 1}, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(
                    f"Failed to embed image after {max_retries} attempts: {e}")
                raise e

# ==================== DATA PROCESSING FUNCTIONS ====================


def process_posts(filename: str) -> Dict[int, Dict[str, Any]]:
    """Load and group posts by topic"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            posts_data = json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found")
        return {}

    topics = {}
    for post in posts_data:
        topic_id = post["topic_id"]
        if topic_id not in topics:
            topics[topic_id] = {
                "topic_title": post.get("topic_title", ""),
                "posts": []
            }
        topics[topic_id]["posts"].append(post)

    # Sort posts by post_number
    for topic in topics.values():
        topic["posts"].sort(key=lambda p: p["post_number"])

    return topics


def build_thread_map(posts: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Build reply tree structure"""
    thread_map = {}
    for post in posts:
        parent = post.get("reply_to_post_number")
        if parent not in thread_map:
            thread_map[parent] = []
        thread_map[parent].append(post)
    return thread_map


def extract_thread(root_num: int, posts: List[Dict[str, Any]], thread_map: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Extract full thread starting from root post"""
    thread = []

    def collect_replies(post_num):
        try:
            post = next(p for p in posts if p["post_number"] == post_num)
            thread.append(post)
            for reply in thread_map.get(post_num, []):
                collect_replies(reply["post_number"])
        except StopIteration:
            pass

    collect_replies(root_num)
    return thread


def process_markdown_files(markdown_dir="markdown_files"):
    """Enhanced markdown processing with better chunking"""
    markdown_data = []

    if not os.path.exists(markdown_dir):
        print(f"Markdown directory '{markdown_dir}' not found.")
        return markdown_data

    for md_file in Path(markdown_dir).glob("*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract frontmatter metadata
            title, original_url, downloaded_at = "", "", ""
            frontmatter_match = re.match(
                r'^---\n(.*?)\n---\n', content, re.DOTALL)
            if frontmatter_match:
                frontmatter = frontmatter_match.group(1)

                title_match = re.search(
                    r'title: ["\']?(.*?)["\']?$', frontmatter, re.MULTILINE)
                if title_match:
                    title = title_match.group(1)

                url_match = re.search(
                    r'original_url: ["\']?(.*?)["\']?$', frontmatter, re.MULTILINE)
                if url_match:
                    original_url = url_match.group(1)

                date_match = re.search(
                    r'downloaded_at: ["\']?(.*?)["\']?$', frontmatter, re.MULTILINE)
                if date_match:
                    downloaded_at = date_match.group(1)

                content = re.sub(r'^---\n.*?\n---\n', '',
                                 content, flags=re.DOTALL)

            chunks = create_markdown_chunks(content, title)

            for i, chunk in enumerate(chunks):
                markdown_data.append({
                    "id": f"md_{md_file.stem}_{i}",
                    "title": title or md_file.stem,
                    "original_url": original_url,
                    "downloaded_at": downloaded_at,
                    "chunk_index": i,
                    "content": chunk,
                    "source_type": "markdown"
                })

        except Exception as e:
            print(f"Error processing {md_file}: {e}")

    return markdown_data


def create_markdown_chunks(content, title, chunk_size=1000, chunk_overlap=200):
    """Enhanced chunking with better overlap handling"""
    if not content:
        return []

    content = re.sub(r'\n+', '\n', content).strip()
    if title:
        content = f"Document: {title}\n\n{content}"

    chunks = []
    sections = re.split(r'\n(#{1,3}\s+.*?)\n', content)

    current_chunk = ""
    current_header = ""

    for section in sections:
        if re.match(r'^#{1,3}\s+', section):
            current_header = section
        else:
            section_content = f"{current_header}\n{section}" if current_header else section

            if len(current_chunk) + len(section_content) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = section_content
            else:
                current_chunk += f"\n\n{section_content}" if current_chunk else section_content

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Apply intelligent overlap
    if len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]

            if len(prev_chunk) > chunk_overlap:
                overlap_start = max(0, len(prev_chunk) - chunk_overlap)
                overlap = prev_chunk[overlap_start:]
                current_chunk = f"{overlap}\n\n{current_chunk}"

            overlapped_chunks.append(current_chunk)
        return overlapped_chunks

    return chunks

# ==================== INDEXING FUNCTIONS ====================


def embed_and_index_threads(topics: Dict[int, Dict[str, Any]], batch_size: int = 50):
    """Enhanced thread indexing with better error handling"""
    vectors = []
    total_processed = 0

    for topic_id, topic_data in tqdm(topics.items(), desc="Processing discourse topics"):
        posts = topic_data["posts"]
        topic_title = topic_data["topic_title"]
        thread_map = build_thread_map(posts)

        root_posts = thread_map.get(None, [])
        for root_post in root_posts:
            try:
                thread = extract_thread(
                    root_post["post_number"], posts, thread_map)

                combined_text = f"Topic: {topic_title}\n\n"
                combined_text += "\n\n---\n\n".join(
                    post["content"].strip() for post in thread if post.get("content")
                )

                if len(combined_text) > 8000:
                    combined_text = combined_text[:8000] + "..."

                embedding = embed_with_retry(combined_text)

                vector = {
                    "id": f"discourse_{topic_id}_{root_post['post_number']}",
                    "values": embedding,
                    "metadata": {
                        "source_type": "discourse",
                        "topic_id": int(topic_id),
                        "topic_title": str(topic_title),
                        "root_post_number": int(float(root_post["post_number"])),
                        "post_numbers": [str(int(float(p["post_number"]))) for p in thread],
                        "combined_text": str(combined_text[:1000])
                    }
                }
                vectors.append(vector)
                total_processed += 1

                if len(vectors) >= batch_size:
                    try:
                        index.upsert(vectors=vectors)
                        print(
                            f"Uploaded discourse batch of {len(vectors)} vectors")
                        vectors = []
                    except Exception as e:
                        print(f"Error upserting discourse vectors: {e}")
                        raise

                time.sleep(0.1)

            except Exception as e:
                print(
                    f"Error processing thread {topic_id}_{root_post['post_number']}: {e}")
                continue

    if vectors:
        try:
            index.upsert(vectors=vectors)
            print(f"Uploaded final discourse batch of {len(vectors)} vectors")
        except Exception as e:
            print(f"Error upserting final discourse vectors: {e}")

    print(f"Total discourse threads processed: {total_processed}")


def embed_and_index_markdown(markdown_data, batch_size=50):
    """Enhanced markdown indexing"""
    vectors = []
    total_processed = 0

    for item in tqdm(markdown_data, desc="Processing markdown"):
        try:
            embedding = embed_with_retry(item["content"])

            vector = {
                "id": item["id"],
                "values": embedding,
                "metadata": {
                    "source_type": "markdown",
                    "title": str(item["title"]),
                    "original_url": str(item.get("original_url", "")),
                    "downloaded_at": str(item.get("downloaded_at", "")),
                    "chunk_index": int(item["chunk_index"]),
                    "combined_text": str(item["content"][:1000])
                }
            }
            vectors.append(vector)
            total_processed += 1

            if len(vectors) >= batch_size:
                try:
                    index.upsert(vectors=vectors)
                    print(f"Uploaded markdown batch of {len(vectors)} vectors")
                    vectors = []
                except Exception as e:
                    print(f"Error upserting markdown vectors: {e}")
                    raise

            time.sleep(0.1)

        except Exception as e:
            print(f"Error processing markdown item {item['id']}: {e}")
            continue

    if vectors:
        try:
            index.upsert(vectors=vectors)
            print(f"Uploaded final markdown batch of {len(vectors)} vectors")
        except Exception as e:
            print(f"Error upserting final markdown vectors: {e}")

    print(f"Total markdown chunks processed: {total_processed}")

# ==================== SEARCH AND RETRIEVAL ====================


def enhanced_semantic_search(query: str, top_k: int = 5, source_filter=None, image_url=None, image_base64=None) -> List[Dict[str, Any]]:
    """Enhanced search with multimodal support and better error handling"""
    try:
        # Get query embedding
        if image_url or image_base64:
            query_embedding = embed_image_with_retry(
                image_url=image_url, image_base64=image_base64)
        else:
            query_embedding = embed_with_retry(query)

        # Build filter
        filter_dict = {}
        if source_filter:
            filter_dict["source_type"] = {"$eq": source_filter}

        # Perform search with retry
        max_search_retries = 3
        for attempt in range(max_search_retries):
            try:
                search_response = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict if filter_dict else None
                )
                break
            except Exception as search_error:
                if attempt < max_search_retries - 1:
                    wait_time = 2 ** attempt
                    print(
                        f"Search error on attempt {attempt + 1}, retrying in {wait_time}s: {search_error}")
                    time.sleep(wait_time)
                else:
                    raise search_error

        # Process results
        results = []
        for match in search_response.matches:
            result = {
                "score": match.score,
                "source_type": match.metadata.get("source_type", "discourse"),
                "combined_text": match.metadata.get("combined_text", "")
            }

            source_type = match.metadata.get("source_type", "discourse")
            if source_type == "markdown":
                result.update({
                    "title": match.metadata.get("title", ""),
                    "original_url": match.metadata.get("original_url", ""),
                    "chunk_index": match.metadata.get("chunk_index", 0)
                })
            elif source_type == "image":
                result.update({
                    "title": match.metadata.get("title", ""),
                    "image_url": match.metadata.get("image_url", ""),
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


def generate_answer(query: str, context_texts: List[str]) -> str:
    """Enhanced answer generation with better context handling"""
    try:
        # Filter out empty contexts
        context_texts = [ctx for ctx in context_texts if ctx and ctx.strip()]
        if not context_texts:
            return "No relevant context found to answer the question."

        context = "\n\n---\n\n".join(context_texts)
        if len(context) > 4000:
            context = context[:4000] + "..."

        messages = [
            {"role": "system",
             "content": "You are a helpful assistant that answers questions based on forum discussions, course materials, and images. Provide clear, accurate answers based on the given context. If the information comes from different sources (forum discussions, course materials, images), mention this in your response. Be concise but comprehensive."},
            {"role": "user", "content": f"Based on these excerpts:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)
                    print(
                        f"Answer generation error on attempt {attempt + 1}, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise e

    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer due to an error."

# ==================== UTILITY FUNCTIONS ====================


def test_connection():
    """Enhanced connection testing"""
    print("Testing connections...")

    # Test OpenAI proxy connection
    try:
        response = openai_client.embeddings.create(
            input="test connection",
            model="text-embedding-3-small"
        )
        print("✅ OpenAI proxy connection successful!")
        openai_success = True
    except Exception as e:
        print(f"❌ OpenAI proxy connection failed: {e}")
        openai_success = False

    # Test Pinecone connection
    try:
        stats = index.describe_index_stats()
        print(
            f"✅ Pinecone connection successful! Index has {stats.total_vector_count} vectors")
        pinecone_success = True
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        pinecone_success = False

    return openai_success and pinecone_success


def interactive_query():
    """Enhanced interactive query interface"""
    print("\n" + "="*60)
    print("🚀 Enhanced RAG System - Interactive Query Mode")
    print("Commands:")
    print("  - Type your question normally")
    print("  - 'discourse:' prefix to search only forum discussions")
    print("  - 'markdown:' prefix to search only course materials")
    print("  - 'image:URL' to search with image from URL")
    print("  - 'quit' to exit")
    print("="*60)

    while True:
        query = input("\n🔍 Enter your question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if not query:
            continue

        # Parse input for filters and images
        source_filter = None
        image_url = None

        if query.startswith("discourse:"):
            source_filter = "discourse"
            query = query[10:].strip()
        elif query.startswith("markdown:"):
            source_filter = "markdown"
            query = query[9:].strip()
        elif query.startswith("image:"):
            parts = query[6:].split(" ", 1)
            if len(parts) == 2:
                image_url = parts[0]
                query = parts[1]
            else:
                print("❌ Invalid image format. Use: image:URL your question")
                continue

        print(f"\n🔎 Searching for: {query}")
        if source_filter:
            print(f"📁 Source filter: {source_filter}")
        if image_url:
            print(f"🖼️  Image URL: {image_url}")

        try:
            results = enhanced_semantic_search(
                query, top_k=3, source_filter=source_filter, image_url=image_url
            )

            if results:
                print("\n📊 Top search results:")
                for i, res in enumerate(results, 1):
                    print(
                        f"\n[{i}] 📈 Score: {res['score']:.4f} | 📂 Source: {res['source_type']}")
                    if res['source_type'] == 'markdown':
                        print(f"📄 Document: {res.get('title', 'Unknown')}")
                        if res.get('original_url'):
                            print(f"🔗 URL: {res['original_url']}")
                    else:
                        print(f"💬 Topic: {res.get('topic_title', 'Unknown')}")
                    print(f"📝 Content: {res['combined_text'][:200]}...")

                # Generate answer
                context_texts = [res["combined_text"] for res in results]
                answer = generate_answer(query, context_texts)
                print("\n🤖 Generated Answer:")
                print("=" * 60)
                print(answer)
                print("=" * 60)
            else:
                print("❌ No relevant results found.")

        except Exception as e:
            print(f"❌ Error during search: {e}")


def initialize_system():
    """Initialize the RAG system for API use"""
    return test_connection()


# Export functions for API use
__all__ = [
    'enhanced_semantic_search',
    'generate_answer',
    'embed_with_retry',
    'embed_image_with_retry',
    'test_connection',
    'initialize_system',
    'process_posts',
    'process_markdown_files',
    'embed_and_index_threads',
    'embed_and_index_markdown'
]

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Test connections first
    if not test_connection():
        print("❌ Connection tests failed. Please check your API keys and configuration.")
        exit(1)

    # Load and process discourse data
    try:
        topics = process_posts("discourse_posts.json")
        print(f"✅ Loaded {len(topics)} discourse topics")
    except Exception as e:
        print(f"⚠️  Error loading discourse data: {e}")
        topics = {}

    # Load and process markdown data
    markdown_data = process_markdown_files("markdown_files")
    print(f"✅ Loaded {len(markdown_data)} markdown chunks")

    # Check existing index data
    try:
        stats = index.describe_index_stats()
        current_vector_count = stats.total_vector_count
        if current_vector_count > 0:
            print(f"📊 Index already contains {current_vector_count} vectors.")
            add_data = input(
                "➕ Add new data to existing index? (y/n): ").lower() == 'y'
        else:
            add_data = True
    except:
        add_data = True

    # Index data if needed
    if add_data:
        if topics:
            print("🔄 Starting discourse indexing...")
            embed_and_index_threads(topics)
            print("✅ Discourse indexing complete")

        if markdown_data:
            print("🔄 Starting markdown indexing...")
            embed_and_index_markdown(markdown_data)
            print("✅ Markdown indexing complete")

    # Final stats
    try:
        final_stats = index.describe_index_stats()
        print(
            f"\n📊 Final index stats: {final_stats.total_vector_count} total vectors")
    except:
        pass

    # Start interactive mode
    interactive_query()
