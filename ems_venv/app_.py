# --- agent_runners.py (Corrected - Use ainvoke_tool) ---

import os
import json
from typing import List, Dict, Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

# --- 1. Define ONE MCP Client Config ---
SERVER_URL = os.getenv("ALL_TOOLS_URL", "http://localhost:8000/mcp") # Default port
try:
    consolidated_client = MultiServerMCPClient({
        "Coordinator_Tools": {"transport": "streamable_http", "url": SERVER_URL},
        "RAG_and_Web_Agent": {"transport": "streamable_http", "url": SERVER_URL},
        "Deep_Analysis_Agent": {"transport": "streamable_http", "url": SERVER_URL},
        "Fact_Checking_Agent": {"transport": "streamable_http", "url": SERVER_URL}
    })
    print(f"[Agent Runners] Configured client for consolidated server on {SERVER_URL}.")
except Exception as e:
    print(f"[Agent Runners] ERROR configuring consolidated client: {e}")
    consolidated_client = None

# --- 2. Define Wrapper Functions (Use ainvoke_tool) ---
@tool
async def call_agent_2_semantic_search(query: str, k: int = 5) -> Any:
    """ Calls RAG Agent: semantic search. """
    if not consolidated_client: return "Error: Client down."
    print(f"[Runner] Calling RAG.semantic_search_tool")
    try:
        # --- CORRECTED METHOD ---
        response = await consolidated_client.ainvoke_tool(
            "RAG_and_Web_Agent", # Namespace
            "semantic_search_tool", # Tool name
            query=query, k=k # Arguments as kwargs
        )
        return response
    except Exception as e: return f"Error calling RAG Semantic Search: {e}"

@tool
async def call_agent_2_keyword_search(query: str, limit: int = 5) -> Any:
    """ Calls RAG Agent: keyword search. """
    if not consolidated_client: return "Error: Client down."
    print(f"[Runner] Calling RAG.keyword_search_tool")
    try:
        # --- CORRECTED METHOD ---
        response = await consolidated_client.ainvoke_tool(
            "RAG_and_Web_Agent",
            "keyword_search_tool",
            query=query, limit=limit
        )
        return response
    except Exception as e: return f"Error calling RAG Keyword Search: {e}"

@tool
async def call_agent_2_web_ingest(urls: List[str] | None = None) -> str:
    """ Calls RAG Agent: web ingest. """
    if not consolidated_client: return "Error: Client down."
    print(f"[Runner] Calling RAG.web_scrape_and_ingest")
    try:
        # --- CORRECTED METHOD ---
        response = await consolidated_client.ainvoke_tool(
            "RAG_and_Web_Agent",
            "web_scrape_and_ingest",
            urls=urls
        )
        return str(response)
    except Exception as e: return f"Error calling RAG Web Ingest: {e}"

@tool
async def call_agent_3_analysis(tool_name: str, **kwargs: Any) -> str:
    """ Calls Analysis Agent: specified tool. """
    if not consolidated_client: return "Error: Client down."
    print(f"[Runner] Calling Analysis.{tool_name}")
    try:
        # --- CORRECTED METHOD ---
        response = await consolidated_client.ainvoke_tool(
            "Deep_Analysis_Agent", # Namespace
            tool_name,             # Tool name
            **kwargs               # Arguments
        )
        return str(response)
    except Exception as e: return f"Error calling Analysis {tool_name}: {e}"

@tool
async def call_agent_4_validation(tool_name: str, **kwargs: Any) -> Any:
    """ Calls Validation Agent: specified tool. """
    if not consolidated_client: return {"error": "Client down."}
    print(f"[Runner] Calling Validation.{tool_name}")
    try:
        # --- CORRECTED METHOD ---
        response = await consolidated_client.ainvoke_tool(
            "Fact_Checking_Agent", # Namespace
            tool_name,             # Tool name
            **kwargs               # Arguments
        )
        return response
    except Exception as e: return {"error": f"Error calling Validation {tool_name}: {e}"}

# --- (List definition and getter function remain the same) ---
agent_runner_tools = [ call_agent_2_semantic_search, call_agent_2_keyword_search, call_agent_2_web_ingest, call_agent_3_analysis, call_agent_4_validation]
def get_agent_runner_tools() -> List[Any]: return agent_runner_tools

if __name__ == '__main__':
    print("Agent runner tools defined for consolidated server (default port). Import 'get_agent_runner_tools'.")
