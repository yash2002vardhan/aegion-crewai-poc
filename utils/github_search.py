# utils/github_search.py
import os
from dotenv import load_dotenv
from crewai_tools import GithubSearchTool

load_dotenv()

# Initialize the GitHub Search Tool with your GitHub token
# You can configure it in different ways:

# 1. Search all repositories with a given content
# github_search_tool = GithubSearchTool(
#     github_repo=None,  # Set to None to search all repositories
#     gh_token=os.getenv("GITHUB_TOKEN"),  # Your GitHub token from .env
#     content_types=['code', 'issue', 'pr'],  # Search in code, issues, and PRs
# )

# 2. Search a specific repository
github_search_tool_specific = GithubSearchTool(
    github_repo="elizaOS/eliza",  # Specific repository
    gh_token=os.getenv("GITHUB_TOKEN"),
    content_types=['code', 'pr'],
)

# 3. Search a specific organization's repositories
# github_search_tool_org = GithubSearchTool(
#     github_repo="org:organization-name",  # All repos in an organization
#     gh_token=os.getenv("GITHUB_TOKEN"),
#     content_types=['code', 'issue'],
# )
