import os
from dotenv import load_dotenv
from crewai_tools import GithubSearchTool
from crewai.tools import tool

load_dotenv()

# Base GitHub tool instance
_base_github_tool = GithubSearchTool(
    github_repo="elizaOS/eliza",
    gh_token=os.getenv("GITHUB_TOKEN"),
    content_types=['code', 'pr','issue'],
)

@tool("Search GitHub Repository")
def github_search_tool(query: str) -> str:
    """Search GitHub repository for code, issues, and pull requests.
    
    USE THIS TOOL FOR:
    - Finding specific code implementations or functions
    - Searching for similar code patterns or examples
    - Looking up issues related to a specific topic
    - Finding pull requests that address specific features or bugs
    - Discovering how certain features are implemented
    - Researching code documentation and comments
    - Finding usage examples of specific APIs or libraries
    - Investigating bug fixes or feature implementations
    
    This tool searches through the configured GitHub repository to find
    relevant code, issues, and pull requests based on your query.
    
    Args:
        query: The search query to find in the GitHub repository
        
    Returns:
        Relevant code snippets, issues, or pull requests from the repository
    """
    return _base_github_tool._run(query)


if __name__ == "__main__":
    # Test the GitHub search integration
    result = github_search_tool(
        query="authentication implementation"
    )
    print(result)
