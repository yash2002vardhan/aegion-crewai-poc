# GitHub Search Tool Integration

This document explains how to use the GitHub Search Tool integrated into the CrewAI CSM Agent.

## Overview

The GitHub Search Tool allows your AI agent to search through GitHub repositories, issues, and pull requests. This is particularly useful for:
- Finding code examples and implementations
- Searching for bug reports or feature requests
- Looking up documentation in repositories
- Investigating similar issues in public repositories

## Setup

### 1. Environment Variables

Add your GitHub token to your `.env` file:

```bash
GITHUB_TOKEN=your_github_personal_access_token_here
```

To create a GitHub Personal Access Token:
1. Go to GitHub Settings > Developer Settings > Personal Access Tokens > Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a descriptive name (e.g., "CrewAI GitHub Search")
4. Select scopes: `repo` (for private repos) or `public_repo` (for public repos only)
5. Click "Generate token" and copy the token to your `.env` file

### 2. Configuration

The GitHub Search Tool is configured in `utils/github_search.py`. You can customize it for different use cases:

#### Search All Repositories
```python
github_search_tool = GithubSearchTool(
    github_repo=None,  # Search across all accessible repositories
    gh_token=os.getenv("GITHUB_TOKEN"),
    content_types=['code', 'issue', 'pr'],
)
```

#### Search Specific Repository
```python
github_search_tool = GithubSearchTool(
    github_repo="owner/repo-name",  # e.g., "anthropics/anthropic-sdk-python"
    gh_token=os.getenv("GITHUB_TOKEN"),
    content_types=['code'],
)
```

#### Search Organization Repositories
```python
github_search_tool = GithubSearchTool(
    github_repo="org:organization-name",  # e.g., "org:anthropics"
    gh_token=os.getenv("GITHUB_TOKEN"),
    content_types=['code', 'issue'],
)
```

### 3. Content Types

You can specify what types of GitHub content to search:
- `'code'` - Search in code files
- `'issue'` - Search in issues
- `'pr'` - Search in pull requests

## Usage

The GitHub Search Tool is already integrated into your CSM agent in `main.py`. The agent can autonomously use it to search for information.

### Example Queries

Here are some example questions your agent can now handle:

1. **Code Search**
   - "Find examples of FastAPI authentication implementation in our repositories"
   - "Search for how we handle database migrations in the codebase"

2. **Issue Search**
   - "Are there any open issues related to memory leaks?"
   - "Find similar bug reports about Qdrant connection timeouts"

3. **Pull Request Search**
   - "Show me recent PRs related to the Slack integration"
   - "Find PRs that modified the telegram utilities"

### Agent Behavior

The agent will automatically use the GitHub Search Tool when:
- A user asks about code examples or implementations
- Technical questions require searching through repositories
- You need to find related issues or documentation
- Investigating bug reports or feature requests

## Advanced Usage

### Custom Search Tool Instance

If you need a specialized GitHub search tool for specific repositories, you can create additional instances:

```python
# In utils/github_search.py
documentation_search = GithubSearchTool(
    github_repo="your-org/documentation",
    gh_token=os.getenv("GITHUB_TOKEN"),
    content_types=['code'],
)

# Then add it to your agent's tools in main.py
tools=[kb_tool, telegram_tool, slack_tool, github_search_tool, documentation_search]
```

### Search Syntax

The GitHub Search Tool uses GitHub's search syntax. Some useful operators:

- `language:python` - Search only Python files
- `path:utils/` - Search in specific directory
- `repo:owner/name` - Search specific repository
- `is:open` - Only open issues/PRs
- `is:closed` - Only closed issues/PRs
- `author:username` - Filter by author

Example: `"authentication error" language:python path:utils/`

## Troubleshooting

### Authentication Errors
- Ensure your `GITHUB_TOKEN` is set correctly in `.env`
- Verify your token has the necessary scopes
- Check if your token has expired

### Rate Limiting
- GitHub API has rate limits (5,000 requests/hour for authenticated users)
- The tool will return rate limit errors if exceeded
- Consider implementing caching for frequently searched queries

### No Results
- Check if you have access to the repositories you're searching
- Verify the search query syntax is correct
- Try broadening your search terms

## Resources

- [CrewAI GitHub Search Tool Documentation](https://docs.crewai.com/en/tools/search-research/githubsearchtool)
- [GitHub Search Syntax](https://docs.github.com/en/search-github/getting-started-with-searching-on-github/understanding-the-search-syntax)
- [GitHub API Rate Limits](https://docs.github.com/en/rest/rate-limit)
