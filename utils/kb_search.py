import os
from crewai_tools import QdrantVectorSearchTool
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Base KB tool instance
_base_kb_tool = QdrantVectorSearchTool(
    qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY", None),
    collection_name=os.getenv("KB_COLLECTION", "customer_success_memory")
)

@tool("Search Knowledge Base")
def kb_search_tool(query: str) -> str:
    """Search the general knowledge base for customer support information.
    
    USE THIS TOOL FOR:
    - General customer support questions
    - Customer history and previous conversations
    - Common support questions and FAQs
    - Account-related inquiries
    - Past issues and resolutions
    - General product information
    - Support policies and procedures
    
    This tool searches historical customer interactions and general support knowledge
    to help answer customer questions based on past experiences and common issues.
    
    Args:
        query: The search query to find in the knowledge base
        
    Returns:
        Relevant information from the knowledge base
    """
    return _base_kb_tool._run(query)

