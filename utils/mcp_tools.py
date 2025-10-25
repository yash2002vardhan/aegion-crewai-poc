"""
MCP Server Tools Integration using MCPServerAdapter

This module provides tools to interact with the custom MCP server for:
1. Fetching documents (general context from messages and documents)
2. Fetching Google Docs (specific documentation search)
3. Fetching Notion (semantic search through Notion content embeddings)

Uses CrewAI's MCPServerAdapter for proper MCP protocol integration.
"""

import os
from dotenv import load_dotenv
from crewai_tools import MCPServerAdapter
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# MCP Server Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://44.220.155.233:8001/mcp")

print(f"🔧 Initializing MCP Server Adapter for: {MCP_SERVER_URL}")
logger.info(f"Initializing MCP Server Adapter for: {MCP_SERVER_URL}")

# Configure server parameters for streamable-http transport
server_params = {
    "url": MCP_SERVER_URL,
    "transport": "streamable-http"
}

# Initialize the adapter and start the connection
# Based on official docs: https://docs.crewai.com/en/mcp/streamable-http
print(f"📡 Connecting to MCP server...")

try:
    mcp_adapter = MCPServerAdapter(server_params)
    
    # Try to access tools directly first (some versions may auto-initialize)
    print(f"🔍 Discovering available tools...")
    try:
        available_tools = mcp_adapter.tools
    except (AttributeError, RuntimeError):
        # If tools not available, try starting the connection
        print(f"⚙️  Starting MCP connection...")
        mcp_adapter.start()
        available_tools = mcp_adapter.tools
except Exception as e:
    print(f"❌ Failed to connect to MCP server: {e}")
    print(f"   Error type: {type(e).__name__}")
    raise

print(f"✅ MCP Server connected successfully")
print(f"📋 Available tools ({len(available_tools)}): {[tool.name for tool in available_tools]}")
logger.info(f"✅ MCP Server connected successfully")
logger.info(f"📋 Available tools: {[tool.name for tool in available_tools]}")

# Extract specific tools we need
# The tools should be named "fetch_documents", "fetch_google_docs", and "fetch_notion" on the MCP server
fetch_documents_tool = None
fetch_google_docs_tool = None
fetch_notion_tool = None

for tool in available_tools:
    tool_name = getattr(tool, 'name', str(tool))
    print(f"  - Found tool: {tool_name}")
    
    if 'fetch_notion' in tool_name.lower():
        fetch_notion_tool = tool
        print(f"    ✅ Mapped to 'fetch_notion_tool'")
        logger.info(f"✅ Found 'fetch_notion' tool: {tool_name}")
    elif 'fetch_documents' in tool_name.lower() or 'retrieve' in tool_name.lower():
        fetch_documents_tool = tool
        print(f"    ✅ Mapped to 'fetch_documents_tool'")
        logger.info(f"✅ Found 'fetch_documents' tool: {tool_name}")
    elif 'google' in tool_name.lower() or 'docs' in tool_name.lower():
        fetch_google_docs_tool = tool
        print(f"    ✅ Mapped to 'fetch_google_docs_tool'")
        logger.info(f"✅ Found 'fetch_google_docs' tool: {tool_name}")

# If tools not found with exact names, assign available tools
if not fetch_documents_tool and len(available_tools) > 0:
    print(f"⚠️  'fetch_documents' not found by name matching, using first available tool")
    fetch_documents_tool = available_tools[0]

if not fetch_google_docs_tool and len(available_tools) > 1:
    print(f"⚠️  'fetch_google_docs' not found by name matching, using second available tool")
    fetch_google_docs_tool = available_tools[1]
elif not fetch_google_docs_tool and len(available_tools) > 0:
    print(f"⚠️  'fetch_google_docs' not found by name matching, reusing first tool")
    fetch_google_docs_tool = available_tools[0]

if not fetch_notion_tool and len(available_tools) > 2:
    print(f"⚠️  'fetch_notion' not found by name matching, using third available tool")
    fetch_notion_tool = available_tools[2]
elif not fetch_notion_tool and len(available_tools) > 0:
    print(f"⚠️  'fetch_notion' not found by name matching, reusing first tool")
    fetch_notion_tool = available_tools[0]

if not fetch_documents_tool or not fetch_google_docs_tool or not fetch_notion_tool:
    error_msg = f"Required tools not found on MCP server. Available tools: {[getattr(t, 'name', str(t)) for t in available_tools]}"
    print(f"❌ {error_msg}")
    logger.error(error_msg)
    raise Exception(error_msg)

print(f"✅ All MCP tools configured successfully")
print(f"   - fetch_documents_tool: {getattr(fetch_documents_tool, 'name', 'unknown')}")
print(f"   - fetch_google_docs_tool: {getattr(fetch_google_docs_tool, 'name', 'unknown')}")
print(f"   - fetch_notion_tool: {getattr(fetch_notion_tool, 'name', 'unknown')}")
logger.info("✅ All MCP tools configured successfully")

# Export the tools and adapter
__all__ = ['fetch_documents_tool', 'fetch_google_docs_tool', 'fetch_notion_tool', 'mcp_adapter']

