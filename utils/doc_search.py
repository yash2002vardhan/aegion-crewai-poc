from crewai_tools import CodeDocsSearchTool
from crewai.tools import tool

# Base tool instance
_base_docs_tool = CodeDocsSearchTool(docs_url='https://flow-docs.mira.network/documentation/get-started/introduction')

@tool("Search Product Documentation")
def code_docs_tool(query: str) -> str:
    """Search official product documentation for technical information.
    
    USE THIS TOOL FOR:
    - Product features and functionality questions
    - Technical how-to questions
    - API documentation and examples
    - Integration guides and setup instructions
    - Configuration and implementation details
    - Product capabilities and specifications
    - Technical tutorials and walkthroughs
    - Code examples and best practices
    
    This tool searches the official product documentation to provide
    accurate, up-to-date technical information about the product.
    
    Args:
        query: The search query to find in product documentation
        
    Returns:
        Relevant documentation content or search results
    """
    return _base_docs_tool._run(query)
