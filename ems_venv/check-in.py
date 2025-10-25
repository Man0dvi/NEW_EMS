# .env

# PostgreSQL Vector DB connection string
PG_CONNECTION_STRING="postgresql://vectoruser:vectorpass@localhost:5433/vectordb"

# OpenAI API Key for Embeddings and LLM
OPENAI_API_KEY="sk-..."

# For the LangChain Agent model (e.g., Gemini-2.5-flash or gpt-4o-mini)
# The LangChain agent needs an LLM to decide which tool to call.
# Use the appropriate key for your chosen provider.
# GOOGLE_API_KEY="AIza..."


# db.py

import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector

load_dotenv()

# --- Configuration ---
PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "deep_research_kb"

if not PG_CONNECTION_STRING:
    raise ValueError("PG_CONNECTION_STRING not set in .env")

# Global Embeddings Object (Used for both indexing and searching)
EMBEDDINGS = OpenAIEmbeddings(
    model="text-embedding-3-large", 
    openai_api_key=OPENAI_API_KEY
)

def get_vector_store() -> PGVector:
    """Initializes and returns the PGVector store object."""
    try:
        store = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=PG_CONNECTION_STRING,
            embedding_function=EMBEDDINGS,
        )
        print(f"[DB] Connected to PGVector collection: {COLLECTION_NAME}")
        return store
    except Exception as e:
        print(f"[DB] ERROR: Could not connect to PGVector: {e}")
        raise

# Global Vector Store Instance
VECTOR_STORE = get_vector_store()


# server.py

import os
import uuid
import asyncio
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# --- FastMCP Setup ---
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# --- LangChain/DB Components ---
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Import shared DB component
from db import VECTOR_STORE, EMBEDDINGS

# Load environment variables
load_dotenv()

# --- Pydantic Schemas for FastMCP (Defines tool I/O) ---
class ScrapeInput(BaseModel):
    web_paths: List[str] = Field(description="A list of URLs to scrape.")

class SearchQuery(BaseModel):
    query: str = Field(description="The user's query or sub-task for searching.")
    k: int = Field(default=5, description="The number of top results to retrieve.")

class RAGResult(BaseModel):
    content: str = Field(description="The retrieved text content of the chunk.")
    source: str = Field(description="The original URL source of the document.")
    # Note: PGVector handles relevance score internally; we simplify the output structure.


# --- FastMCP Initialization ---
mcp = FastMCP(
    name="Agent2_RAG_Server", 
    instructions="Provides web scraping, chunking, and semantic search over a PostgreSQL Vector DB.",
)

# --- Tool 1: Web Scraper ---
@mcp.tool()
async def web_scraper_tool(input: ScrapeInput) -> List[Dict[str, str]]:
    """
    Scrapes raw text content from a list of URLs using selective HTML parsing.

    Args:
        input: Pydantic model containing the list of web_paths (URLs).
            
    Returns:
        A list of dictionaries, each with 'text' and 'url', for further processing.
    """
    print(f"[WebTool] Starting scrape for {len(input.web_paths)} path(s)...")
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content", "content", "main"))
    results = []
    
    for path in input.web_paths:
        try:
            # WebBaseLoader is synchronous, so we run it in a thread pool
            loader = WebBaseLoader(web_paths=[path], bs_kwargs={"parse_only": bs4_strainer})
            docs = await asyncio.to_thread(loader.load)
            
            if docs:
                results.append({"text": docs[0].page_content, "url": path})
                print(f"[WebTool] Scraped from: {path} (Chars: {len(docs[0].page_content)})")
        except Exception as e:
            print(f"[WebTool] ERROR scraping {path}: {e}")
            
    return results

# --- Tool 2: RAG Pipeline (Chunking & Storage) ---
@mcp.tool()
async def rag_pipeline_tool(scraped_documents: List[Dict[str, str]]) -> str:
    """
    Splits raw text documents, embeds the chunks, and stores them in the PGVector database.

    Args:
        scraped_documents: List of dictionaries with 'text' and 'url' (output from web_scraper_tool).

    Returns:
        A status message indicating the outcome of the storage operation.
    """
    if not scraped_documents:
        return "No documents provided to the RAG pipeline. Skipping storage."
        
    print(f"[RAGTool] Starting chunking and storage for {len(scraped_documents)} document(s)...")

    # 1. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    # Convert dictionaries to LangChain Document objects
    lc_documents = []
    for doc in scraped_documents:
        lc_documents.append(
            Document(page_content=doc["text"], metadata={"source": doc["url"], "timestamp": datetime.now().isoformat()})
        )
    
    chunks = text_splitter.split_documents(lc_documents)
    print(f"[RAGTool] Total chunks created: {len(chunks)}")

    # 2. Embedding and Storage (PGVector handles this via add_documents)
    try:
        # add_documents is often synchronous, wrap for a safe async environment
        await asyncio.to_thread(VECTOR_STORE.add_documents, documents=chunks)
        return f"Successfully stored {len(chunks)} chunks in PGVector."
    except Exception as e:
        return f"ERROR storing chunks in PGVector: {e}"

# --- Tool 3: Semantic Search Tool ---
@mcp.tool()
async def semantic_search_tool(input: SearchQuery) -> List[RAGResult]:
    """
    Performs a semantic similarity search against the vectorized knowledge base 
    to retrieve the top 'k' most relevant chunks.

    Args:
        input: Pydantic model containing the query string and number of results (k).

    Returns:
        A list of RAGResult models (chunks and metadata).
    """
    print(f"[SearchTool] Performing semantic search for: '{input.query}' (k={input.k})")
    
    try:
        # similarity_search is synchronous, wrap for async execution
        retrieved_docs = await asyncio.to_thread(
            VECTOR_STORE.similarity_search, 
            query=input.query, 
            k=input.k
        )
        
        results = [
            RAGResult(
                content=doc.page_content,
                source=doc.metadata.get("source", "N/A"),
                # Relevance score omitted for simplicity, but can be added via similarity_search_with_score
            )
            for doc in retrieved_docs
        ]
            
        print(f"[SearchTool] Retrieved {len(results)} chunks.")
        return results
    
    except Exception as e:
        print(f"[SearchTool] ERROR during semantic search: {e}")
        return []

if __name__ == "__main__":
    print("Starting Agent 2 FastMCP Server...")
    mcp.run() # Starts the FastMCP server


# agent.py

import asyncio
from typing import List
from fastmcp import Client
from langchain.agents import create_agent
from langchain.tools import tool, Tool

# --- LLM for the Agent's Reasoning ---
from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field

# --- Configuration ---
MCP_SERVER_URL = "http://127.0.0.1:8000" 
LLM_MODEL = "gpt-4o-mini" # A fast, capable model for tool-calling/reasoning

# Example URL and query for the RAG task
TEST_URLS = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
TEST_QUERY = "What are the core loop steps and memory types used in a successful LLM powered autonomous agent?"

# --- Tools exposed by the FastMCP Server (Wrapped for LangChain) ---
# We use the @tool decorator on wrapper functions to expose them to the LangChain Agent
# and handle the necessary data marshalling via the MCP Client.

# Global Client Instance (Initialized in main)
MCP_CLIENT: Client = None

class ScrapeSchema(BaseModel):
    """Input for the web_scraper_tool."""
    web_paths: List[str] = Field(description="A list of URLs to scrape.")

@tool(args_schema=ScrapeSchema)
def fetch_documents(web_paths: List[str]) -> str:
    """Fetches raw text content from the given URLs and returns it as a stringified list."""
    if not MCP_CLIENT: return "Error: MCP Client not initialized."
    # The client handles the MCP call to the server's web_scraper_tool
    raw_docs = asyncio.run(MCP_CLIENT.web_scraper_tool(input={"web_paths": web_paths}))
    
    # We return a string summary for the LLM to read and decide the next step
    if not raw_docs:
        return "Web scraping failed: No documents were retrieved."
    
    # Step 2: Immediately trigger storage (simulates the RAG chain logic of Agent 1)
    print("\n--- Agent Logic: Auto-triggering RAG Storage (Step 2) ---")
    storage_status = asyncio.run(MCP_CLIENT.rag_pipeline_tool(
        scraped_documents=raw_docs, 
        source_url=web_paths[0] # Simplification: passing the main URL
    ))
    
    return f"Documents scraped (Chars: {len(raw_docs[0]['text'])}). Storage Status: {storage_status}"

class SearchSchema(BaseModel):
    """Input for the semantic_search_tool."""
    query: str = Field(description="The final research query for RAG retrieval.")
    k: int = Field(default=5, description="The number of top chunks to retrieve.")

@tool(args_schema=SearchSchema)
def retrieve_relevant_context(query: str, k: int = 5) -> str:
    """Performs semantic search on the PGVector DB and returns the top relevant chunks."""
    if not MCP_CLIENT: return "Error: MCP Client not initialized."
    
    # Calls the server's semantic_search_tool
    results = asyncio.run(MCP_CLIENT.semantic_search_tool(input={"query": query, "k": k}))
    
    if not results:
        return "Semantic search returned no relevant context chunks."

    # Format output for the LLM to synthesize the final answer
    formatted_context = []
    for i, result in enumerate(results):
        formatted_context.append(f"CHUNK {i+1} (Source: {result.source}): {result.content}")
        
    return "\n---\n".join(formatted_context)

# --- Main Agent Orchestration ---
async def main():
    global MCP_CLIENT
    
    # 1. Initialize MCP Client (The communication bridge)
    MCP_CLIENT = Client(url=MCP_SERVER_URL)
    
    # 2. Check Server Connection and Get LangChain Tools
    try:
        await MCP_CLIENT.get_server_info()
        print(f"Connected to FastMCP server at {MCP_SERVER_URL}.")
    except Exception as e:
        print(f"ERROR: Could not connect to FastMCP server: {e}. Ensure 'server.py' is running.")
        return

    # 3. Define the LangChain Agent
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
    
    # The tools available to the LangChain Agent (the wrappers we just created)
    tools = [fetch_documents, retrieve_relevant_context]
    
    # Use LangChain's prebuilt agent constructor (LangGraph-based)
    agent_executor = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are the Research Coordinator Agent (Agent 1). Your goal is to conduct a deep research "
            "task. First, use 'fetch_documents' to get and store the data. Then, use "
            "'retrieve_relevant_context' with the final query to get the key information."
        )
    )

    # 4. Invoke the Agent (The full RAG execution)
    print("\n--- Running LangChain Agent Workflow ---")
    
    # The Agent must decide to call fetch_documents first, then retrieve_relevant_context.
    
    # Initial message to kick off the workflow.
    full_message = (
        f"Start research workflow. 1. Scrape the URL: {TEST_URLS[0]}. "
        f"2. Then answer the query: '{TEST_QUERY}'."
    )
    
    # The agent state will manage the multi-step reasoning
    result = await agent_executor.invoke(
        {"messages": [HumanMessage(content=full_message)]}
    )

    # 5. Output Result
    print("\n--- Final Agent Output (Input for Agent 3) ---")
    print(f"Agent Final Answer:\n{result['messages'][-1].content}")
    print("\nWorkflow successful: Data was scraped, stored in PGVector, and retrieved by the LangChain agent.")

if __name__ == "__main__":
    asyncio.run(main())
