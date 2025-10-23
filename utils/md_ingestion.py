"""
Markdown File Ingestion and Hybrid Search for Milvus using LlamaIndex

This utility provides functionality to:
1. Ingest markdown files into Milvus vector database
2. Perform hybrid search (semantic + BM25) using LlamaIndex
3. Clean scraped documentation and preserve markdown structure
4. Handle tables, code blocks, and section hierarchies
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from crewai.tools import tool

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
MILVUS_URI = os.getenv("MD_MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION = os.getenv("MD_MILVUS_COLLECTION", "markdown_docs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536  # Dimension for text-embedding-3-small

# Chunking parameters
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Patterns for cleaning scraped documentation
NAVIGATION_PATTERNS = [
    r'\[Skip to main content\].*?\n',
    r'\[.*?home page.*?\].*?\n',
    r'Search documentation\.\.\.\n',
    r'Ctrl [A-Z]\n',
    r'Navigation\n',
    r'Search\.\.\.\n',
    r'On this page\n',
    r'Was this page helpful\?\n',
    r'YesNo\n',
    r'\[Powered by .*?\]\(.*?\)\n',
    r'Assistant\n',
    r'Responses are generated using AI.*?\.\n',
]


class MarkdownIngestion:
    """Handle markdown file ingestion into Milvus with hybrid search support using LlamaIndex."""
    
    def __init__(
        self,
        milvus_uri: str = MILVUS_URI,
        collection_name: str = MILVUS_COLLECTION,
        embedding_model: str = EMBEDDING_MODEL,
        embedding_dim: int = EMBEDDING_DIM,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        """
        Initialize the Markdown Ingestion system with LlamaIndex.
        
        Args:
            milvus_uri: Milvus connection URI
            collection_name: Name of the collection to use
            embedding_model: OpenAI embedding model name
            embedding_dim: Dimension of the embedding vectors
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Configure LlamaIndex settings
        Settings.embed_model = OpenAIEmbedding(
            model=embedding_model,
            api_key=OPENAI_API_KEY
        )
        Settings.llm = OpenAI(
            model="gpt-4",
            api_key=OPENAI_API_KEY
        )
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        
        # Initialize Milvus vector store with hybrid search enabled
        self.vector_store = MilvusVectorStore(
            uri=self.milvus_uri,
            dim=self.embedding_dim,
            collection_name=self.collection_name,
            enable_sparse=True,  # Enable BM25 full-text search
            hybrid_ranker="WeightedRanker",
            hybrid_ranker_params={"weights": [0.7, 0.3]},  # 70% semantic, 30% BM25
            overwrite=False  # Don't overwrite existing collection by default
        )
        
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        logger.info(f"Initialized MilvusVectorStore with collection '{collection_name}'")
        logger.info(f"Hybrid search enabled: 70% semantic + 30% BM25")
    
    def _clean_scraped_documentation(self, content: str) -> str:
        """
        Clean scraped documentation by removing navigation and boilerplate.
        
        Args:
            content: Raw markdown content
            
        Returns:
            Cleaned markdown content
        """
        cleaned = content
        
        # Remove common navigation patterns
        for pattern in NAVIGATION_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        
        # Remove repeated navigation links (common in scraped docs)
        cleaned = re.sub(r'(\[.*?\]\(.*?\)\s*){5,}', '', cleaned)
        
        # Remove excessive newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # Remove lines that are just "---" or "===" (separators)
        cleaned = re.sub(r'^[=-]{3,}$', '', cleaned, flags=re.MULTILINE)
        
        return cleaned.strip()
    
    def _extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """
        Extract metadata from markdown content.
        
        Args:
            file_path: Path to the file
            content: Markdown content
            
        Returns:
            Dictionary with metadata
        """
        # Extract headings
        heading_pattern = r'^#{1,6}\s+(.+)$'
        headings = re.findall(heading_pattern, content, re.MULTILINE)
        
        # Clean headings
        cleaned_headings = []
        for heading in headings:
            # Remove markdown links
            heading = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', heading)
            # Remove emojis and special markers
            heading = re.sub(r'[⚡🛠️🎯🧠⚙️🔄🔗🔀📝📊]+', '', heading).strip()
            if heading:
                cleaned_headings.append(heading)
        
        # Check for code blocks and tables
        has_code = bool(re.search(r'```', content))
        has_table = bool(re.search(r'\|.*\|', content))
        
        return {
            "file_name": file_path.name,
            "file_path": str(file_path.absolute()),
            "headings": cleaned_headings,
            "primary_heading": cleaned_headings[0] if cleaned_headings else file_path.stem,
            "has_code": has_code,
            "has_table": has_table,
            "doc_type": "documentation"
        }
    
    def _load_and_clean_documents(
        self,
        file_path: str,
        clean_scraped: bool = True
    ) -> List[Document]:
        """
        Load markdown file and create LlamaIndex documents with metadata.
        
        Args:
            file_path: Path to markdown file or directory
            clean_scraped: Whether to clean scraped documentation
            
        Returns:
            List of LlamaIndex Document objects
        """
        path = Path(file_path)
        
        if path.is_file():
            # Load single file
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Clean if requested
            original_size = len(content)
            if clean_scraped:
                content = self._clean_scraped_documentation(content)
                logger.info(f"Cleaned content: {original_size} -> {len(content)} chars")
            
            # Extract metadata
            metadata = self._extract_metadata(path, content)
            
            # Create LlamaIndex Document
            doc = Document(
                text=content,
                metadata=metadata
            )
            
            return [doc]
        
        elif path.is_dir():
            # Load directory using SimpleDirectoryReader
            documents = []
            md_files = list(path.glob("**/*.md"))
            
            logger.info(f"Found {len(md_files)} markdown files")
            
            for md_file in md_files:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Clean if requested
                if clean_scraped:
                    content = self._clean_scraped_documentation(content)
                
                # Extract metadata
                metadata = self._extract_metadata(md_file, content)
                
                # Create LlamaIndex Document
                doc = Document(
                    text=content,
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
        
        else:
            raise ValueError(f"Invalid path: {file_path}")
    
    def ingest(
        self,
        path: str,
        clean_scraped: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest markdown files into Milvus using LlamaIndex.
        
        Args:
            path: Path to markdown file or directory
            clean_scraped: Whether to clean scraped documentation
            show_progress: Whether to show progress
            
        Returns:
            Dictionary with ingestion statistics
        """
        try:
            logger.info(f"Loading documents from: {path}")
            
            # Load and clean documents
            documents = self._load_and_clean_documents(path, clean_scraped)
            
            if not documents:
                logger.warning("No documents loaded")
                return {
                    "status": "no_documents",
                    "documents_loaded": 0,
                    "chunks_created": 0
                }
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Create index with hybrid search
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                show_progress=show_progress
            )
            
            # Get stats
            stats = {
                "status": "success",
                "documents_loaded": len(documents),
                "collection": self.collection_name,
                "files": [doc.metadata.get("file_name") for doc in documents]
            }
            
            logger.info(f"✅ Successfully ingested {len(documents)} documents")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {e}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword/BM25 search (0-1)
            
        Returns:
            List of search results with metadata
        """
        try:
            # Update ranker weights if different from initialization
            if semantic_weight != 0.7 or keyword_weight != 0.3:
                self.vector_store = MilvusVectorStore(
                    uri=self.milvus_uri,
                    dim=self.embedding_dim,
                    collection_name=self.collection_name,
                    enable_sparse=True,
                    hybrid_ranker="WeightedRanker",
                    hybrid_ranker_params={"weights": [semantic_weight, keyword_weight]},
                    overwrite=False
                )
                self.storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store
                )
            
            # Load index from existing vector store
            index = VectorStoreIndex.from_vector_store(self.vector_store)
            
            # Create query engine with hybrid search
            query_engine = index.as_query_engine(
                vector_store_query_mode="hybrid",
                similarity_top_k=top_k
            )
            
            # Execute query
            response = query_engine.query(query)
            
            # Format results
            results = []
            if hasattr(response, 'source_nodes'):
                for idx, node in enumerate(response.source_nodes):
                    result = {
                        "rank": idx + 1,
                        "score": node.score if hasattr(node, 'score') else 0.0,
                        "text": node.text,
                        "metadata": node.metadata if hasattr(node, 'metadata') else {}
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            raise
    
    def query_with_answer(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> Dict[str, Any]:
        """
        Perform hybrid search query and generate an answer using LLM.
        
        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic search
            keyword_weight: Weight for keyword search
            
        Returns:
            Dictionary with answer and source nodes
        """
        try:
            # Update ranker weights if needed
            if semantic_weight != 0.7 or keyword_weight != 0.3:
                self.vector_store = MilvusVectorStore(
                    uri=self.milvus_uri,
                    dim=self.embedding_dim,
                    collection_name=self.collection_name,
                    enable_sparse=True,
                    hybrid_ranker="WeightedRanker",
                    hybrid_ranker_params={"weights": [semantic_weight, keyword_weight]},
                    overwrite=False
                )
                self.storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store
                )
            
            # Load index
            index = VectorStoreIndex.from_vector_store(self.vector_store)
            
            # Create query engine
            query_engine = index.as_query_engine(
                vector_store_query_mode="hybrid",
                similarity_top_k=top_k
            )
            
            # Execute query
            response = query_engine.query(query)
            
            # Format response
            result = {
                "query": query,
                "answer": str(response),
                "sources": []
            }
            
            if hasattr(response, 'source_nodes'):
                for idx, node in enumerate(response.source_nodes):
                    source = {
                        "rank": idx + 1,
                        "score": node.score if hasattr(node, 'score') else 0.0,
                        "text": node.text[:300] + "..." if len(node.text) > 300 else node.text,
                        "metadata": node.metadata if hasattr(node, 'metadata') else {}
                    }
                    result["sources"].append(source)
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying with answer: {e}")
            raise
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.vector_store.client.drop_collection(collection_name=self.collection_name)
            logger.info(f"✅ Collection '{self.collection_name}' deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            stats = self.vector_store.client.get_collection_stats(
                collection_name=self.collection_name
            )
            return {
                "collection_name": self.collection_name,
                "stats": stats
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise


# Global instance for the tool
_md_search_instance = None

def get_md_search_instance():
    """Get or create the global MarkdownIngestion instance."""
    global _md_search_instance
    if _md_search_instance is None:
        _md_search_instance = MarkdownIngestion()
    return _md_search_instance


@tool("Search Markdown Documentation")
def markdown_docs_search_tool(query: str) -> str:
    """Search through markdown documentation stored in Milvus using hybrid search (semantic + BM25).
    
    USE THIS TOOL FOR:
    - Questions about technical documentation
    - How-to guides and tutorials
    - API documentation and code examples
    - Configuration and setup instructions
    - Product features and specifications
    - Troubleshooting and best practices
    - Any questions about documented topics in markdown files
    
    This tool uses hybrid search combining:
    - 70% semantic search (understands meaning and context)
    - 30% BM25 keyword search (exact term matching)
    
    This provides the best of both worlds for accurate retrieval.
    
    Args:
        query: The search query to find in the documentation
        
    Returns:
        Relevant documentation content with LLM-generated answer and sources
    """
    try:
        # Get the markdown search instance
        md_system = get_md_search_instance()
        
        # Perform hybrid search with answer generation
        result = md_system.query_with_answer(
            query=query,
            top_k=5,
            semantic_weight=0.7,
            keyword_weight=0.3
        )
        
        # Format the response
        response_parts = [
            f"📚 Answer: {result['answer']}",
            ""
        ]
        
        if result.get('sources'):
            response_parts.append(f"📖 Sources ({len(result['sources'])}):")
            response_parts.append("")
            
            for source in result['sources'][:3]:  # Show top 3 sources
                metadata = source.get('metadata', {})
                file_name = metadata.get('file_name', 'Unknown file')
                heading = metadata.get('primary_heading', '')
                
                response_parts.append(f"• {file_name}")
                if heading:
                    response_parts.append(f"  Section: {heading}")
                response_parts.append(f"  Score: {source['score']:.3f}")
                response_parts.append(f"  Content: {source['text']}")
                response_parts.append("")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Error in markdown_docs_search_tool: {e}")
        return f"Error searching documentation: {str(e)}. Please check if documents have been ingested into the collection."


# CLI Interface
if __name__ == "__main__":
    import argparse
    import textwrap
    
    parser = argparse.ArgumentParser(
        description="Markdown Ingestion and Hybrid Search for Milvus using LlamaIndex"
    )
    parser.add_argument(
        "action",
        choices=["ingest", "search", "query", "stats", "delete"],
        help="Action to perform"
    )
    parser.add_argument("--path", help="Path to markdown file or directory")
    parser.add_argument("--query", help="Search query (uses hybrid search: semantic + BM25)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of search results")
    parser.add_argument("--semantic-weight", type=float, default=0.7, help="Weight for semantic search (0-1)")
    parser.add_argument("--keyword-weight", type=float, default=0.3, help="Weight for keyword search (0-1)")
    parser.add_argument("--no-clean", action="store_true", help="Don't clean scraped documentation")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size for text splitting")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="Overlap between chunks")
    
    args = parser.parse_args()
    
    # Initialize the system
    md_system = MarkdownIngestion(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap
    )
    
    if args.action == "ingest":
        if not args.path:
            print("❌ Error: --path is required for ingestion")
            exit(1)
        
        print(f"📥 Ingesting documents from: {args.path}\n")
        stats = md_system.ingest(
            path=args.path,
            clean_scraped=not args.no_clean
        )
        
        print(f"\n✅ Ingestion complete!")
        print(f"   Documents: {stats['documents_loaded']}")
        print(f"   Collection: {stats['collection']}")
        if stats.get('files'):
            print(f"   Files: {', '.join(stats['files'][:3])}" + 
                  (f" and {len(stats['files'])-3} more" if len(stats['files']) > 3 else ""))
    
    elif args.action == "search":
        if not args.query:
            print("❌ Error: --query is required for search")
            exit(1)
        
        print(f"🔍 Searching: '{args.query}'")
        print(f"   Hybrid weights: {args.semantic_weight:.1f} semantic + {args.keyword_weight:.1f} keyword\n")
        
        results = md_system.search(
            query=args.query,
            top_k=args.top_k,
            semantic_weight=args.semantic_weight,
            keyword_weight=args.keyword_weight
        )
        
        if results:
            print(f"Found {len(results)} results:\n")
            for result in results:
                print(f"[{result['rank']}] Score: {result['score']:.4f}")
                if result['metadata'].get('file_name'):
                    print(f"    📄 File: {result['metadata']['file_name']}")
                if result['metadata'].get('primary_heading'):
                    print(f"    📌 Section: {result['metadata']['primary_heading']}")
                if result['metadata'].get('has_code'):
                    print("    📝 [Contains Code]", end="")
                if result['metadata'].get('has_table'):
                    print(" 📊 [Contains Table]", end="")
                if result['metadata'].get('has_code') or result['metadata'].get('has_table'):
                    print()
                print(f"    Preview: {result['text'][:200]}...")
                print()
        else:
            print("No results found.")
    
    elif args.action == "query":
        if not args.query:
            print("❌ Error: --query is required for query")
            exit(1)
        
        print(f"🔍 Querying: '{args.query}'")
        print(f"   Hybrid weights: {args.semantic_weight:.1f} semantic + {args.keyword_weight:.1f} keyword\n")
        
        result = md_system.query_with_answer(
            query=args.query,
            top_k=args.top_k,
            semantic_weight=args.semantic_weight,
            keyword_weight=args.keyword_weight
        )
        
        print("💡 Answer:")
        print(textwrap.fill(result['answer'], width=100))
        
        if result['sources']:
            print(f"\n📚 Sources ({len(result['sources'])}):\n")
            for source in result['sources']:
                print(f"[{source['rank']}] Score: {source['score']:.4f}")
                if source['metadata'].get('file_name'):
                    print(f"    📄 {source['metadata']['file_name']}")
                print(f"    {source['text']}")
                print()
    
    elif args.action == "stats":
        stats = md_system.get_stats()
        print(f"📊 Collection: {stats['collection_name']}")
        print(f"   Stats: {stats['stats']}")
    
    elif args.action == "delete":
        confirm = input(f"⚠️  Are you sure you want to delete collection '{md_system.collection_name}'? (yes/no): ")
        if confirm.lower() == "yes":
            md_system.delete_collection()
            print("✅ Collection deleted successfully")
        else:
            print("❌ Deletion cancelled")
