# --- agent_runners.py (Refactored for Single Server) ---

import os
import json
from typing import List, Dict, Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

# --- 1. Define ONE MCP Client for the Consolidated Server ---
# All agents' tools are now on port 8000 under different paths/namespaces
# We define clients for each namespace pointing to the same base URL.
# The namespace in the client definition *must match* the namespace used in FastMCP()
SERVER_BASE_URL = os.getenv("ALL_TOOLS_URL", "http://localhost:8000")

try:
    consolidated_client = MultiServerMCPClient({
        # Namespace must match FastMCP("Coordinator_Tools")
        "Coordinator_Tools": {
            "transport": "streamable_http",
            "url": f"{SERVER_BASE_URL}/coordinator/mcp" # URL includes the mount path + /mcp
        },
        # Namespace must match FastMCP("RAG_and_Web_Agent")
        "RAG_and_Web_Agent": {
            "transport": "streamable_http",
            "url": f"{SERVER_BASE_URL}/rag/mcp"
        },
        # Namespace must match FastMCP("Deep_Analysis_Agent")
        "Deep_Analysis_Agent": {
            "transport": "streamable_http",
            "url": f"{SERVER_BASE_URL}/analysis/mcp"
        },
        # Namespace must match FastMCP("Fact_Checking_Agent")
        "Fact_Checking_Agent": {
            "transport": "streamable_http",
            "url": f"{SERVER_BASE_URL}/validation/mcp"
        }
    })
    print("[Agent Runners] Configured client for consolidated server on port 8000.")
except Exception as e:
    print(f"[Agent Runners] ERROR configuring consolidated client: {e}")
    consolidated_client = None


# --- 2. Define Wrapper Functions as Tools (Calls use NAMESPACE.TOOL_NAME) ---

# --- Agent 2 Tools ---
@tool
async def call_agent_2_semantic_search(query: str, k: int = 5) -> Any:
    """Calls the RAG Agent to perform semantic search."""
    if not consolidated_client: return "Error: Consolidated client not available."
    print(f"[Agent Runners] Calling RAG: semantic_search_tool")
    try:
        # Use NAMESPACE.TOOL_NAME format
        response = await consolidated_client.call(
            "RAG_and_Web_Agent.semantic_search_tool",
            query=query, k=k
        )
        return response
    except Exception as e: return f"Error calling RAG Semantic Search: {e}"

@tool
async def call_agent_2_keyword_search(query: str, limit: int = 5) -> Any:
    """Calls the RAG Agent to perform keyword search."""
    if not consolidated_client: return "Error: Consolidated client not available."
    print(f"[Agent Runners] Calling RAG: keyword_search_tool")
    try:
        response = await consolidated_client.call(
            "RAG_and_Web_Agent.keyword_search_tool",
            query=query, limit=limit
        )
        return response
    except Exception as e: return f"Error calling RAG Keyword Search: {e}"

@tool
async def call_agent_2_web_ingest(urls: List[str] | None = None) -> str:
    """Calls the RAG Agent to scrape and ingest content."""
    if not consolidated_client: return "Error: Consolidated client not available."
    print(f"[Agent Runners] Calling RAG: web_scrape_and_ingest")
    try:
        response = await consolidated_client.call(
            "RAG_and_Web_Agent.web_scrape_and_ingest",
            urls=urls
        )
        return str(response) # Ensure string return
    except Exception as e: return f"Error calling RAG Web Ingest: {e}"

# --- Agent 3 Tools ---
@tool
async def call_agent_3_analysis(tool_name: str, **kwargs: Any) -> str:
    """Calls the Deep Analysis Agent to perform complex reasoning."""
    if not consolidated_client: return "Error: Consolidated client not available."
    print(f"[Agent Runners] Calling Analysis: {tool_name}")
    try:
        response = await consolidated_client.call(
            f"Deep_Analysis_Agent.{tool_name}", # Use NAMESPACE.TOOL_NAME
            **kwargs
        )
        if not isinstance(response, str): response = json.dumps(response, indent=2)
        return response
    except Exception as e: return f"Error calling Analysis ({tool_name}): {e}"

# --- Agent 4 Tools ---
@tool
async def call_agent_4_validation(tool_name: str, **kwargs: Any) -> Any:
    """Calls the Fact-Checking Agent to perform validation tasks."""
    if not consolidated_client: return {"error": "Consolidated client not available."}
    print(f"[Agent Runners] Calling Validation: {tool_name}")
    try:
        response = await consolidated_client.call(
            f"Fact_Checking_Agent.{tool_name}", # Use NAMESPACE.TOOL_NAME
            **kwargs
        )
        return response
    except Exception as e: return {"error": f"Error calling Validation ({tool_name}): {e}"}

# --- List of all agent runner tools available to the Coordinator ---
agent_runner_tools = [
    call_agent_2_semantic_search,
    call_agent_2_keyword_search,
    call_agent_2_web_ingest,
    call_agent_3_analysis,
    call_agent_4_validation,
]

def get_agent_runner_tools() -> List[Any]:
    return agent_runner_tools

if __name__ == '__main__':
    print("Agent runner tools defined for consolidated server. Import 'get_agent_runner_tools'.")
