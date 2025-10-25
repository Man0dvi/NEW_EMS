# agent.py

import asyncio
from typing import List, Dict, Any
from fastmcp import Client
from langchain.agents import create_agent
from langchain.tools import tool, Tool
from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field

# --- Configuration ---
MCP_SERVER_URL = "http://127.0.0.1:8000" 
LLM_MODEL = "gpt-4o-mini" # LLM for Agent 1's reasoning
TEST_URLS = ["http://lilianweng.github.io/posts/2023-06-23-agent/"]
TEST_QUERY = "What are the core loop steps and memory types used in a successful LLM powered autonomous agent?"

# Global Client Instance
MCP_CLIENT: Client = None

# --- Agent Tool Definitions (Wrappers for MCP Calls) ---

class ScrapeSchema(BaseModel): web_paths: List[str] = Field(description="A list of URLs to scrape.")
class SearchSchema(BaseModel): query: str = Field(description="The research query for RAG retrieval.")

@tool(args_schema=ScrapeSchema)
def fetch_and_store_data(web_paths: List[str]) -> str:
    """
    1. Fetches raw text from URLs. 2. Triggers the RAG pipeline to chunk, 
    extract keywords, embed, and store data in PGVector.
    """
    if not MCP_CLIENT: return "Error: MCP Client not initialized."
    
    # 1. Call Scraper
    raw_docs = asyncio.run(MCP_CLIENT.web_scraper_tool(input={"web_paths": web_paths}))
    if not raw_docs: return "Web scraping failed."
    
    # 2. Call RAG Pipeline for Storage
    storage_status = asyncio.run(MCP_CLIENT.rag_pipeline_tool(
        scraped_documents=raw_docs, 
        source_url=web_paths[0]
    ))
    
    return f"Documents stored. Status: {storage_status}"


@tool(args_schema=SearchSchema)
def retrieve_and_synthesize_context(query: str) -> str:
    """
    Performs hybrid semantic and keyword search, de-duplicates results, 
    and synthesizes the context for the final LLM response.
    """
    if not MCP_CLIENT: return "Error: MCP Client not initialized."
    
    # 1. Call Semantic Search
    semantic_results = asyncio.run(MCP_CLIENT.semantic_search_tool(input={"query": query, "k": 3}))
    
    # 2. Call Keyword Search
    keyword_results = asyncio.run(MCP_CLIENT.keyword_search_tool(input={"query": query, "k": 3}))

    # 3. Hybrid Synthesis: Combine and De-duplicate
    combined_results = {}
    
    # Use content as the key for de-duplication
    for res in semantic_results + keyword_results:
        # Use a simplified de-duplication key
        dedupe_key = res.content[:50] 
        if dedupe_key not in combined_results:
            combined_results[dedupe_key] = res

    # 4. Format Output for Final LLM (Agent 1 Synthesis Tool)
    formatted_context = []
    for i, res in enumerate(combined_results.values()):
        formatted_context.append(
            f"CHUNK {i+1} [Type: {res.relevance_type}, Source: {res.source}]: {res.content}"
        )
        
    return f"CONSOLIDATED CONTEXT:\n{'-'*20}\n" + "\n".join(formatted_context)


async def main():
    global MCP_CLIENT
    MCP_CLIENT = Client(url=MCP_SERVER_URL)
    
    try:
        await MCP_CLIENT.get_server_info()
    except Exception:
        print("ERROR: Could not connect to FastMCP server. Ensure 'server.py' is running.")
        return

    # 1. Define the LangChain Agent
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
    tools = [fetch_and_store_data, retrieve_and_synthesize_context]
    
    agent_executor = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are the Research Coordinator Agent (Agent 1). Your goal is to fully answer the user's research query. "
            "First, use the 'fetch_and_store_data' tool to acquire and process the knowledge. "
            "Then, use the 'retrieve_and_synthesize_context' tool to gather all relevant information. "
            "Finally, use the gathered context to provide a single, comprehensive final answer."
        )
    )

    # 2. Invoke the Agent
    full_message = (
        f"1. Scrape the URL: {TEST_URLS[0]}. 2. Then, fully answer the query: '{TEST_QUERY}'."
    )
    
    print("\n--- Running LangChain Agent Workflow ---")
    result = await agent_executor.invoke({"messages": [HumanMessage(content=full_message)]})

    # 3. Output Result
    print("\n--- Final Agent Output ---")
    print(f"Agent Final Answer:\n{result['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())
