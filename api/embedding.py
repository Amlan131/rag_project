import json
from typing import List, Dict, Any
import os
import re
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize OpenAI client with IITM proxy
openai_client = OpenAI(
    # Use AIPROXY_TOKEN instead of OPENAI_API_KEY
    api_key=os.getenv("AIPROXY_TOKEN"),
    base_url="https://aiproxy.sanand.workers.dev/openai/v1"  # Use proxy base URL
)

# Initialize Pinecone client
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize Pinecone index with free tier region
index_name = "discourse-embeddings"
if index_name not in pinecone.list_indexes().names():
    try:
        pinecone.create_index(
            name=index_name,
            dimension=1536,  # OpenAI text-embedding-3-small dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Free tier supported region
            )
        )
        print(f"Successfully created index: {index_name}")
    except Exception as e:
        print(f"Error creating index: {e}")
        raise

index = pinecone.Index(index_name)


def process_posts(filename: str) -> Dict[int, Dict[str, Any]]:
    """Load and group posts by topic"""
    with open(filename, "r", encoding="utf-8") as f:
        posts_data = json.load(f)

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
        post = next(p for p in posts if p["post_number"] == post_num)
        thread.append(post)
        for reply in thread_map.get(post_num, []):
            collect_replies(reply["post_number"])

    collect_replies(root_num)
    return thread


def embed_with_retry(text: str, max_retries: int = 3):
    """Get embedding with retry logic and rate limiting"""
    for attempt in range(max_retries):
        try:
            response = openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"API error, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                raise e


def process_markdown_files(markdown_dir="markdown_files"):
    """Process markdown files and add them to your existing system"""
    markdown_data = []

    if not os.path.exists(markdown_dir):
        print(
            f"Markdown directory '{markdown_dir}' not found. Skipping markdown processing.")
        return markdown_data

    for md_file in Path(markdown_dir).glob("*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract frontmatter metadata
            title = ""
            original_url = ""
            downloaded_at = ""

            frontmatter_match = re.match(
                r'^---\n(.*?)\n---\n', content, re.DOTALL)
            if frontmatter_match:
                frontmatter = frontmatter_match.group(1)

                # Extract metadata
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

                # Remove frontmatter from content
                content = re.sub(r'^---\n.*?\n---\n', '',
                                 content, flags=re.DOTALL)

            # Create chunks similar to your discourse processing
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
    """Create intelligent chunks from markdown content"""
    if not content:
        return []

    # Clean up content
    content = re.sub(r'\n+', '\n', content)
    content = content.strip()

    # Add title context
    if title:
        content = f"Document: {title}\n\n{content}"

    chunks = []

    # Split by headers first (## or ###)
    sections = re.split(r'\n(#{1,3}\s+.*?)\n', content)

    current_chunk = ""
    current_header = ""

    for i, section in enumerate(sections):
        if re.match(r'^#{1,3}\s+', section):  # This is a header
            current_header = section
        else:  # This is content
            section_content = f"{current_header}\n{section}" if current_header else section

            # If adding this section exceeds chunk size, save current chunk
            if len(current_chunk) + len(section_content) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = section_content
            else:
                current_chunk += f"\n\n{section_content}" if current_chunk else section_content

    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Apply overlap between chunks
    if len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]

            # Add overlap from previous chunk
            if len(prev_chunk) > chunk_overlap:
                overlap_start = max(0, len(prev_chunk) - chunk_overlap)
                overlap = prev_chunk[overlap_start:]
                current_chunk = f"{overlap}\n\n{current_chunk}"

            overlapped_chunks.append(current_chunk)

        return overlapped_chunks

    return chunks


def embed_and_index_threads(topics: Dict[int, Dict[str, Any]], batch_size: int = 50):
    """Embed threads using OpenAI and index in Pinecone"""
    vectors = []
    total_processed = 0

    for topic_id, topic_data in tqdm(topics.items(), desc="Processing discourse topics"):
        posts = topic_data["posts"]
        topic_title = topic_data["topic_title"]
        thread_map = build_thread_map(posts)

        # Process root posts (those without parents)
        root_posts = thread_map.get(None, [])
        for root_post in root_posts:
            try:
                thread = extract_thread(
                    root_post["post_number"], posts, thread_map)

                # Combine thread text
                combined_text = f"Topic: {topic_title}\n\n"
                combined_text += "\n\n---\n\n".join(
                    post["content"].strip() for post in thread
                )

                # Limit text length to avoid token limits
                if len(combined_text) > 8000:  # Rough token limit
                    combined_text = combined_text[:8000] + "..."

                # Get embedding from OpenAI with retry
                embedding = embed_with_retry(combined_text)

                # Prepare vector for Pinecone with proper data types
                vector = {
                    "id": f"discourse_{topic_id}_{root_post['post_number']}",
                    "values": embedding,
                    "metadata": {
                        "source_type": "discourse",
                        "topic_id": int(topic_id),  # Ensure integer
                        "topic_title": str(topic_title),  # Ensure string
                        # Handle float conversion
                        "root_post_number": int(float(root_post["post_number"])),
                        # Convert floats to strings
                        "post_numbers": [str(int(float(p["post_number"]))) for p in thread],
                        # Ensure string and limit size
                        "combined_text": str(combined_text[:1000])
                    }
                }
                vectors.append(vector)
                total_processed += 1

                # Batch upsert when we have enough vectors
                if len(vectors) >= batch_size:
                    try:
                        index.upsert(vectors=vectors)
                        print(
                            f"Uploaded discourse batch of {len(vectors)} vectors")
                        vectors = []
                    except Exception as e:
                        print(f"Error upserting discourse vectors: {e}")
                        raise

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                print(
                    f"Error processing thread {topic_id}_{root_post['post_number']}: {e}")
                continue

    # Upsert any remaining vectors
    if vectors:
        try:
            index.upsert(vectors=vectors)
            print(f"Uploaded final discourse batch of {len(vectors)} vectors")
        except Exception as e:
            print(f"Error upserting final discourse vectors: {e}")

    print(f"Total discourse threads processed: {total_processed}")


def embed_and_index_markdown(markdown_data, batch_size=50):
    """Add markdown embeddings to your existing Pinecone index"""
    vectors = []
    total_processed = 0

    for item in tqdm(markdown_data, desc="Processing markdown"):
        try:
            # Get embedding
            embedding = embed_with_retry(item["content"])

            # Prepare vector for Pinecone
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

            # Batch upsert
            if len(vectors) >= batch_size:
                try:
                    index.upsert(vectors=vectors)
                    print(f"Uploaded markdown batch of {len(vectors)} vectors")
                    vectors = []
                except Exception as e:
                    print(f"Error upserting markdown vectors: {e}")
                    raise

            time.sleep(0.1)  # Rate limiting

        except Exception as e:
            print(f"Error processing markdown item {item['id']}: {e}")
            continue

    # Upsert remaining vectors
    if vectors:
        try:
            index.upsert(vectors=vectors)
            print(f"Uploaded final markdown batch of {len(vectors)} vectors")
        except Exception as e:
            print(f"Error upserting final markdown vectors: {e}")

    print(f"Total markdown chunks processed: {total_processed}")


def enhanced_semantic_search(query: str, top_k: int = 5, source_filter=None) -> List[Dict[str, Any]]:
    """Enhanced search that can filter by source type"""
    try:
        query_embedding = embed_with_retry(query)

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


def generate_answer(query: str, context_texts: List[str]) -> str:
    """Generate answer using OpenAI"""
    try:
        context = "\n\n---\n\n".join(context_texts)

        # Limit context length
        if len(context) > 4000:
            context = context[:4000] + "..."

        messages = [
            {"role": "system",
                "content": "You are a helpful assistant that answers questions based on forum discussions and course materials. Provide clear, accurate answers based on the given context. If the information comes from different sources (forum discussions vs course materials), mention this in your response."},
            {"role": "user", "content": f"Based on these excerpts:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ]

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer due to an error."


def test_connection():
    """Test API connections"""
    print("Testing connections...")

    # Test OpenAI proxy connection
    try:
        response = openai_client.embeddings.create(
            input="test",
            model="text-embedding-3-small"
        )
        print("✅ OpenAI proxy connection successful!")
    except Exception as e:
        print(f"❌ OpenAI proxy connection failed: {e}")
        return False

    # Test Pinecone connection
    try:
        stats = index.describe_index_stats()
        print(
            f"✅ Pinecone connection successful! Index has {stats.total_vector_count} vectors")
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        return False

    return True


def interactive_query():
    """Interactive query interface"""
    print("\n" + "="*50)
    print("RAG System Ready - Interactive Query Mode")
    print("Commands:")
    print("  - Type your question normally")
    print("  - 'discourse:' prefix to search only forum discussions")
    print("  - 'markdown:' prefix to search only course materials")
    print("  - 'quit' to exit")
    print("="*50)

    while True:
        query = input("\nEnter your question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not query:
            continue

        # Check for source filter
        source_filter = None
        if query.startswith("discourse:"):
            source_filter = "discourse"
            query = query[10:].strip()
        elif query.startswith("markdown:"):
            source_filter = "markdown"
            query = query[9:].strip()

        print(f"\nSearching for: {query}")
        if source_filter:
            print(f"Source filter: {source_filter}")

        results = enhanced_semantic_search(
            query, top_k=3, source_filter=source_filter)

        if results:
            print("\nTop search results:")
            for i, res in enumerate(results, 1):
                print(
                    f"\n[{i}] Score: {res['score']:.4f} | Source: {res['source_type']}")
                if res['source_type'] == 'markdown':
                    print(f"Document: {res.get('title', 'Unknown')}")
                else:
                    print(f"Topic: {res.get('topic_title', 'Unknown')}")
                print(f"Content: {res['combined_text'][:200]}...\n")

            # Generate answer
            context_texts = [res["combined_text"] for res in results]
            answer = generate_answer(query, context_texts)
            print("\nGenerated Answer:")
            print("=" * 50)
            print(answer)
            print("=" * 50)
        else:
            print("No relevant results found.")
# Add this to your existing RAG script


def initialize_system():
    """Initialize the RAG system for API use"""
    global index, openai_client
    # Your existing initialization code
    return test_connection()


# Make sure these functions are available for import
__all__ = [
    'enhanced_semantic_search',
    'generate_answer',
    'embed_with_retry',
    'test_connection',
    'initialize_system'
]


# Example usage
if __name__ == "__main__":
    # Test connections first
    if not test_connection():
        print("Connection tests failed. Please check your API keys and configuration.")
        exit(1)

    # Load and process discourse data
    try:
        topics = process_posts("discourse_posts.json")
        print(f"Loaded {len(topics)} discourse topics")
    except FileNotFoundError:
        print("Error: discourse_posts.json file not found!")
        topics = {}
    except Exception as e:
        print(f"Error loading discourse data: {e}")
        topics = {}

    # Load and process markdown data
    markdown_data = process_markdown_files("markdown_files")
    print(f"Loaded {len(markdown_data)} markdown chunks")

    # Check if index already has data
    try:
        stats = index.describe_index_stats()
        current_vector_count = stats.total_vector_count
        if current_vector_count > 0:
            print(f"Index already contains {current_vector_count} vectors.")
            add_data = input(
                "Add new data to existing index? (y/n): ").lower() == 'y'
        else:
            add_data = True
    except:
        add_data = True

    # Index discourse data
    if add_data and topics:
        print("Starting discourse indexing...")
        embed_and_index_threads(topics)
        print("Discourse indexing complete")

    # Index markdown data
    if add_data and markdown_data:
        print("Starting markdown indexing...")
        embed_and_index_markdown(markdown_data)
        print("Markdown indexing complete")

    # Final stats
    try:
        final_stats = index.describe_index_stats()
        print(
            f"\nFinal index stats: {final_stats.total_vector_count} total vectors")
    except:
        pass

    # Example searches
    print("\n" + "="*50)
    print("Example Searches:")

    # Test search with both sources
    query = "How to submit assignments?"
    print(f"\nExample Query: {query}")
    results = enhanced_semantic_search(query, top_k=3)

    if results:
        print("\nTop search results:")
        for i, res in enumerate(results, 1):
            print(
                f"\n[{i}] Score: {res['score']:.4f} | Source: {res['source_type']}")
            if res['source_type'] == 'markdown':
                print(f"Document: {res.get('title', 'Unknown')}")
            else:
                print(f"Topic: {res.get('topic_title', 'Unknown')}")
            print(f"Content: {res['combined_text'][:300]}...\n")

        # Generate answer
        context_texts = [res["combined_text"] for res in results]
        answer = generate_answer(query, context_texts)
        print("\nGenerated Answer:")
        print("=" * 50)
        print(answer)
        print("=" * 50)
    else:
        print("No relevant results found.")

    # Start interactive mode
    interactive_query()
