import os
from crewai.tools import tool
from dotenv import load_dotenv
from pymilvus import MilvusClient, connections
from typing import List, Dict, Any
import logging
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

# Milvus configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "44.220.155.233")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_ENTITY_COLLECTION = os.getenv("MILVUS_ENTITY_COLLECTION", "entity_index")
MILVUS_MESSAGE_COLLECTION = os.getenv("MILVUS_MESSAGE_COLLECTION", "messages")

# OpenAI client for embeddings
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = os.getenv("MILVUS_EMBEDDING_MODEL", "text-embedding-3-small")

# Initialize Milvus client
def get_milvus_client():
    """Initialize and return Milvus client."""
    try:
        client = MilvusClient(
            uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}"
        )
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        return None


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding vector for the input text using OpenAI.

    Args:
        text: Input text to embed

    Returns:
        List of floats representing the embedding vector
    """
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise


def search_milvus_collection(client: MilvusClient, collection_name: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search a Milvus collection using semantic search.

    Args:
        client: MilvusClient instance
        collection_name: Name of the collection to search
        query: Search query text
        limit: Maximum number of results to return

    Returns:
        List of search results with text and metadata
    """
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query)

        # Perform semantic search with embedding vector
        results = client.search(
            collection_name=collection_name,
            data=[query_embedding],  # Pass embedding vector
            limit=limit,
            output_fields=["*"]  # Return all fields
        )

        # Format results
        formatted_results = []
        if results and len(results) > 0:
            for hit in results[0]:
                formatted_results.append({
                    "id": hit.get("id"),
                    "distance": hit.get("distance"),
                    "text": hit.get("entity", {}).get("text", ""),
                    "metadata": hit.get("entity", {})
                })

        return formatted_results
    except Exception as e:
        logger.error(f"Error searching collection {collection_name}: {e}")
        return []


@tool("Search Mira Documentation")
def mira_docs_tool(query: str) -> str:
    """Search Mira product documentation stored in Milvus vector database.

    USE THIS TOOL FOR:
    - Questions about Mira product features and capabilities
    - Mira technical documentation and specifications
    - How to use Mira functionality
    - Mira API documentation and examples
    - Mira integration guides and setup instructions
    - Mira configuration and implementation details
    - Mira best practices and tutorials
    - Mira workflow and process documentation
    - Any question that mentions "Mira" or relates to Mira product

    This tool searches the official Mira documentation stored in a Milvus
    vector database to provide accurate, up-to-date information about the
    Mira product and its features.

    Args:
        query: The search query to find in Mira documentation

    Returns:
        Relevant Mira documentation content or error message
    """
    try:
        # Get Milvus client
        client = get_milvus_client()
        if not client:
            return "Error: Unable to connect to Milvus database. Please check the connection settings."

        # Search both collections
        entity_results = search_milvus_collection(client, MILVUS_ENTITY_COLLECTION, query, limit=3)
        message_results = search_milvus_collection(client, MILVUS_MESSAGE_COLLECTION, query, limit=3)

        # Combine and format results
        all_results = []

        if entity_results:
            all_results.append("=== Entity Index Results ===")
            for idx, result in enumerate(entity_results, 1):
                text = result.get("text", "No content available")
                distance = result.get("distance", 0)
                all_results.append(f"\n[Result {idx}] (Relevance: {distance:.3f})")
                all_results.append(text)

        if message_results:
            all_results.append("\n\n=== Message Results ===")
            for idx, result in enumerate(message_results, 1):
                text = result.get("text", "No content available")
                distance = result.get("distance", 0)
                all_results.append(f"\n[Result {idx}] (Relevance: {distance:.3f})")
                all_results.append(text)

        if not entity_results and not message_results:
            return f"No results found in Mira documentation for query: '{query}'. The documentation may not contain information about this topic."

        return "\n".join(all_results)

    except Exception as e:
        logger.error(f"Error in mira_docs_tool: {e}")
        return f"Error searching Mira documentation: {str(e)}"
