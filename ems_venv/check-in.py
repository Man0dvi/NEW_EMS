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
from db import VECTOR_STORE # Import the PGVector instance

# --- External Utility for Keyword Extraction (Simulated) ---
# In a real system, you would use NLTK, spaCy, or KeyBERT here.
def extract_keywords_mock(text: str) -> List[str]:
    """Simulates extracting keywords from a text chunk."""
    # A real implementation would use: from rake_nltk import Rake; r = Rake(); r.extract_keywords_from_text(text); return r.get_ranked_phrases()[:5]
    # For simulation, we tokenize and return unique words.
    keywords = [word.lower() for word in text.split() if len(word) > 3][:10]
    return list(set(keywords))

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

# --- Tool 1: Web Scraper (No Change) ---
@mcp.tool()
async def web_scraper_tool(input: ScrapeInput) -> List[Dict[str, str]]:
    """Scrapes raw text content from a list of URLs using selective HTML parsing."""
    # ... [Implementation of web_scraper_tool remains the same] ...
    # Placeholder for brevity: copy logic from previous response
    print(f"[WebTool] Starting scrape for {len(input.web_paths)} path(s)...")
    # ... [Rest of web_scraper_tool logic]
    return [{"text": "Sample text about the core loop steps and memory types...", "url": "http://sample.com/doc1"}] 


# --- Tool 2: RAG Pipeline (UPDATED for Keyword Extraction) ---
@mcp.tool()
async def rag_pipeline_tool(scraped_documents: List[Dict[str, str]]) -> str:
    """
    Splits raw text documents, extracts keywords, embeds, and stores them in the PGVector database.
    """
    if not scraped_documents: return "No documents provided to the RAG pipeline. Skipping storage."
        
    print(f"[RAGTool] Starting chunking, keyword extraction, and storage...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    lc_documents = []
    for doc in scraped_documents:
        # Split documents into chunks
        chunks = text_splitter.split_text(doc["text"])
        
        for chunk_content in chunks:
            # *********** NEW STEP: KEYWORD EXTRACTION ***********
            keywords = extract_keywords_mock(chunk_content)
            
            lc_documents.append(
                Document(
                    page_content=chunk_content, 
                    metadata={
                        "source": doc["url"], 
                        "timestamp": datetime.now().isoformat(),
                        "keywords": keywords # Stored in metadata, mapped to the 'keywords' array column in DB
                    }
                )
            )
    
    print(f"[RAGTool] Total chunks created: {len(lc_documents)}. Storing to DB...")

    # Embedding and Storage (PGVector handles this via add_documents)
    try:
        await asyncio.to_thread(VECTOR_STORE.add_documents, documents=lc_documents)
        return f"Successfully stored {len(lc_documents)} chunks with keywords in PGVector."
    except Exception as e:
        return f"ERROR storing chunks in PGVector: {e}"

# --- Tool 3: Semantic Search Tool (UPDATED Output) ---
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

# --- Tool 4: Keyword Search Tool (NEW) ---
@mcp.tool()
async def keyword_search_tool(input: SearchQuery) -> List[RAGResult]:
    """
    Performs a direct keyword/lexical search against the indexed 'keywords' and 'text_content' 
    fields in the PostgreSQL database.
    """
    print(f"[SearchTool] Performing KEYWORD search for: '{input.query}' (k={input.k})")
    
    # 1. Prepare SQL Query (using raw SQL for specific keyword array/full-text search)
    # NOTE: This requires running raw SQL, which LangChain's PGVector doesn't expose directly.
    # We will simulate the result structure here, but a real implementation requires a DB wrapper 
    # (like SQLAlchemy or a custom function using psycopg2) to execute:
    # SQL: SELECT * FROM deep_research_chunks WHERE keywords @> ARRAY['word1', 'word2'] LIMIT k;
    
    # --- Simulated Keyword Retrieval ---
    # We simulate a highly effective keyword retrieval result
    simulated_results = [
        RAGResult(
            content="The autonomous agent's core steps involve **Observation, Planning, and Action**, utilizing short-term working memory and long-term knowledge.",
            source="http://lilianweng.github.io/posts/2023-06-23-agent/",
            relevance_type="Keyword"
        ),
        RAGResult(
            content="**Memory** is categorized into short-term (context window) and **long-term** (vector store/DB).",
            source="http://lilianweng.github.io/posts/2023-06-23-agent/",
            relevance_type="Keyword"
        )
    ]
    
    # Limit results based on k
    return simulated_results[:input.k]


if __name__ == "__main__":
    print("Starting Agent 2 FastMCP Server...")
    mcp.run()
