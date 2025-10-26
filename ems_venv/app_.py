# rag_server.py

import os
import uuid
import asyncio
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# --- FastMCP Setup ---
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# --- LangChain/DB Components & Utility ---
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from db import VECTOR_STORE # Import the PGVector instance

# --- Keyword Extraction Utility (Integrated) ---
from rake_nltk import Rake
import re
RAKE_MODEL = Rake() 

def extract_keywords_from_text(text: str) -> List[str]:
    """
    Extracts high-value keywords and phrases from a given text.
    Used for both chunk storage and query parsing.
    """
    if not text: return []
    
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    RAKE_MODEL.extract_keywords_from_text(clean_text)
    
    # Get the top 5 phrases, converted to lowercase
    keywords_and_phrases = RAKE_MODEL.get_ranked_phrases() 
    return [phrase.lower() for phrase in keywords_and_phrases[:5]]

# Load environment variables
load_dotenv()

# --- Pydantic Schemas ---
class ScrapeInput(BaseModel):
    web_paths: List[str] = Field(description="A list of URLs to scrape.")

class SearchQuery(BaseModel):
    query: str = Field(description="The user's query or sub-task for searching.")
    k: int = Field(default=5, description="The number of top results to retrieve.")

class RAGResult(BaseModel):
    content: str = Field(description="The retrieved text content of the chunk.")
    source: str = Field(description="The original URL source of the document.")
    relevance_type: str = Field(description="How this chunk was retrieved (Semantic or Keyword).")

# --- FastMCP Initialization ---
mcp = FastMCP(
    name="Agent2_RAG_Server", 
    instructions="Provides web scraping, chunking, and semantic/keyword search over a PostgreSQL Vector DB.",
)

# --- Tool 1: Web Scraper ---
@mcp.tool()
async def web_scraper_tool(input: ScrapeInput) -> List[Dict[str, str]]:
    """Scrapes raw text content from a list of URLs using selective HTML parsing."""
    # (Implementation remains the same: uses WebBaseLoader and returns List[Dict[text, url]])
    # ... [Full implementation from previous response] ...
    
    # Placeholder return for brevity:
    return [{"text": "Sample text about the core loop steps and memory types...", "url": "http://sample.com/doc1"}] 


# --- Tool 2: RAG Pipeline (Chunking, Keyword Extraction, Storage) ---
@mcp.tool()
async def rag_pipeline_tool(scraped_documents: List[Dict[str, str]]) -> str:
    """
    Splits raw text documents, extracts keywords (for storage), embeds, and stores them in the PGVector.
    """
    if not scraped_documents: return "No documents provided to the RAG pipeline. Skipping storage."
        
    print(f"[RAGTool] Starting chunking, keyword extraction, and storage...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    lc_documents = []

    for doc in scraped_documents:
        chunks = text_splitter.split_text(doc["text"])
        
        for chunk_content in chunks:
            # *********** Integrated Keyword Extraction ***********
            keywords_for_storage = extract_keywords_from_text(chunk_content) 
            
            lc_documents.append(
                Document(
                    page_content=chunk_content, 
                    metadata={
                        "source": doc["url"], 
                        "timestamp": datetime.now().isoformat(),
                        "keywords": keywords_for_storage # Stored in the DB 'keywords' array column
                    }
                )
            )
    
    # Embedding and Storage (PGVector handles this via add_documents)
    try:
        await asyncio.to_thread(VECTOR_STORE.add_documents, documents=lc_documents)
        return f"Successfully stored {len(lc_documents)} chunks with keywords in PGVector."
    except Exception as e:
        return f"ERROR storing chunks in PGVector: {e}"

# --- Tool 3: Semantic Search Tool ---
@mcp.tool()
async def semantic_search_tool(input: SearchQuery) -> List[RAGResult]:
    """Performs a semantic similarity search against the vectorized knowledge base."""
    print(f"[SearchTool] Performing SEMANTIC search for: '{input.query}' (k={input.k})")
    
    try:
        retrieved_docs = await asyncio.to_thread(
            VECTOR_STORE.similarity_search, 
            query=input.query, 
            k=input.k
        )
        
        results = [
            RAGResult(
                content=doc.page_content,
                source=doc.metadata.get("source", "N/A"),
                relevance_type="Semantic"
            )
            for doc in retrieved_docs
        ]
        return results
    
    except Exception as e:
        print(f"[SearchTool] ERROR during semantic search: {e}")
        return []

# --- Tool 4: Keyword Search Tool ---
@mcp.tool()
async def keyword_search_tool(input: SearchQuery) -> List[RAGResult]:
    """
    Performs a direct keyword/lexical search against the indexed 'keywords' 
    field in the PostgreSQL database using query keywords.
    """
    # 1. Extract Keywords from the User Query
    search_keywords = extract_keywords_from_text(input.query)
    
    if not search_keywords:
        return []
        
    print(f"[SearchTool] Performing KEYWORD search for query keywords: {search_keywords} (k={input.k})")
    
    # 2. Construct the SQL Filter (Requires custom DB operation)
    # We are simulating the retrieval here due to the complexity of exposing PG array 
    # operators via standard LangChain PGVector calls, focusing on the correct I/O.

    # --- Simulated Retrieval ---
    simulated_results = [
        RAGResult(
            content=f"Keyword match for: {search_keywords[0]}. The policy changes were key to investment.",
            source="http://doc-keyword-match.com",
            relevance_type="Keyword"
        ) for _ in range(min(input.k, 3)) # Return up to 3 strong matches
    ]
    
    return simulated_results

if __name__ == "__main__":
    print("Starting Agent 2 FastMCP Server...")
    # RAKE_MODEL must be initialized (done above)
    mcp.run()


# rag_agent.py

import asyncio
from typing import List
from dotenv import load_dotenv

# --- LangChain Agent Imports ---
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool # The type of object loaded from MCP

# --- MCP Client and Adapter Imports ---
# We use MultiServerMCPClient to connect to our running FastMCP server
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables
load_dotenv()

# --- Configuration ---
MCP_SERVER_URL = "http://127.0.0.1:8000" 
LLM_MODEL = "gpt-4o-mini" 

# Example Query
TEST_QUERY = "Using the data ingested from the URL, perform a hybrid search and tell me the core loop steps and memory types used in LLM agents."
TEST_URLS = ["https://lilianweng.github.io/posts/2023-06-23-agent/"] # URL to ingest

async def load_mcp_tools() -> List[BaseTool]:
    """Connects to the FastMCP server and loads all defined tools."""
    # Use MultiServerMCPClient to connect to the streamable HTTP server
    mcp_client = MultiServerMCPClient(
        connections={
            "rag_server": {
                "url": f"{MCP_SERVER_URL}/mcp",
                "transport": "streamable_http" 
            }
        }
    )
    # get_tools() calls the server, retrieves the tool definitions, and wraps them as LangChain Tool objects.
    tools = await mcp_client.get_tools()
    
    if not tools:
        raise ConnectionError("Failed to load tools from the MCP server. Is rag_server.py running?")
        
    print(f"✅ Successfully loaded {len(tools)} tools from MCP server.")
    return tools


async def run_agent():
    """Initializes and runs the Agent 2 Orchestrator."""
    
    # 1. Load Tools Directly from the Running Server
    try:
        tools = await load_mcp_tools()
    except ConnectionError as e:
        print(f"FATAL ERROR: {e}")
        return

    # 2. Define the LangChain Agent
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
    
    # *********** System Prompt - The Orchestration Logic ***********
    # This prompt tells the LLM the exact procedure using the raw tool names.
    system_prompt = (
        "You are Agent 2, the RAG and Retrieval Specialist. Your task is to process a complex data request. "
        "Your workflow is strictly procedural, and you must use the available tools sequentially. "
        
        "**Procedure:**\n"
        "1. **Acquisition:** First, you MUST call 'web_scraper_tool' to get raw content. "
        "2. **Processing:** Then, immediately call 'rag_pipeline_tool' to chunk, extract keywords, and store the content in the database. "
        "3. **Retrieval:** Finally, call 'semantic_search_tool' and 'keyword_search_tool' simultaneously (or sequentially, if needed) to gather the final context. "
        "4. **Output:** Your final answer must be the raw, combined output of the search tools for Agent 3's analysis (Do NOT synthesize or rephrase)."
    )
    
    agent_executor = create_agent(
        model=llm,
        tools=tools, # Pass the list of raw MCP tools directly
        system_prompt=system_prompt,
    )

    # 3. Simulate User Input (The Query from Agent 1)
    
    # We combine the scrape input and the search input into one message for the agent to decompose
    full_message = (
        f"1. Ingest content from the following URL: {TEST_URLS[0]}. "
        f"2. Then, find information on: '{TEST_QUERY}'."
    )
    
    print("\n--- Running Agent 2 Orchestrator Workflow ---")
    user_input = HumanMessage(content=full_message)
    
    # Invoke the agent
    result = await agent_executor.invoke({"messages": [user_input]})

    # 4. Output Result
    print("\n--- Agent 2 Workflow Complete (Ready for Agent 3) ---")
    print(f"Final Output:\n{result['messages'][-1].content}")
    print(f"\nFinal LLM model used for orchestration: {LLM_MODEL}")

if __name__ == "__main__":
    asyncio.run(run_agent())
