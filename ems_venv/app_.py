# --- agent_2_tools.py ---
# Defines MCP tools for Agent 2: Web Scraper & Document Retrieval Agent
# This version uses a persistent PostgreSQL database with the pgvector extension
# and a custom table schema as requested.

import os
import uuid
import yake  # For keyword extraction
import datetime
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# --- Database Imports ---
# We'll use SQLAlchemy to manage the DB connection and execute raw SQL
# and psycopg2 as the underlying driver.
import sqlalchemy
from sqlalchemy import create_engine, text

# --- LangChain Imports ---
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. --- INITIAL SETUP ---
load_dotenv()

# Initialize the MCP server
mcp = FastMCP("RAG_and_Web_Agent")

# --- 2. DATABASE SETUP ---
# Get the connection string from your specification
PG_CONNECTION_STRING = "postgresql://vectoruser:vectorpass@localhost:5433/vectordb"
TABLE_NAME = "deep_research_chunks"

try:
    # Create the SQLAlchemy engine
    db_engine = create_engine(PG_CONNECTION_STRING)
    print(f"[Agent 2] Connecting to PostgreSQL at {db_engine.url.host}...")
except Exception as e:
    print(f"[Agent 2] ERROR: Could not create DB engine. Is PostgreSQL running? {e}")
    exit(1)

# Initialize the models (same as before)
try:
    embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    # Get embedding dimension
    test_embedding = embeddings_model.embed_query("test")
    EMBEDDING_DIM = len(test_embedding)
    print(f"[Agent 2] OpenAI Embeddings loaded. Dimension: {EMBEDDING_DIM}")
except ImportError:
    raise ImportError("OpenAI provider not found. Please install langchain-openai")
except Exception as e:
    print(f"[Agent 2] ERROR: Could not init OpenAIEmbeddings. Check API key. {e}")
    exit(1)
    
# Initialize keyword extractor
kw_extractor = yake.KeywordExtractor(n=1, top=20, dedupLim=0.9)

def setup_database():
    """
    Ensures the 'vector' extension is enabled and the 'deep_research_chunks'
    table exists with the correct schema.
    """
    try:
        with db_engine.connect() as conn:
            # Step 1: Enable the pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # Step 2: Create the table as per the specified schema
            # We use "IF NOT EXISTS" to make this function idempotent
            table_creation_query = text(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                chunk_id UUID PRIMARY KEY,
                document_id UUID,
                source_url TEXT,
                timestamp TIMESTAMPTZ,
                text_content TEXT,
                keywords TEXT[],
                embedding_vector VECTOR({EMBEDDING_DIM})
            );
            """)
            conn.execute(table_creation_query)
            
            # Step 3: Create an index for vector search (HNSW or IVFFlat)
            # This is crucial for performance.
            conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_hnsw_embedding
            ON {TABLE_NAME}
            USING hnsw (embedding_vector vector_l2_ops);
            """))
            # Commit the transaction
            conn.commit()
        print(f"[Agent 2] Database setup complete. Table '{TABLE_NAME}' is ready.")
    except Exception as e:
        print(f"[Agent 2] ERROR during database setup: {e}")
        print("Please ensure the PostgreSQL user 'vectoruser' has permissions to CREATE.")


# --- 3. MCP TOOL DEFINITIONS ---

@mcp.tool()
def web_scrape_and_ingest(urls: List[str]) -> str:
    """
    Tool 1: Scrapes content from URLs, chunks it, extracts keywords,
    creates embeddings, and stores everything in the PostgreSQL database
    according to the specified 'deep_research_chunks' schema.
    """
    print(f"[Agent 2] Received request to scrape {len(urls)} URLs...")

    # --- LOAD ---
    loader = WebBaseLoader(urls)
    try:
        docs = loader.load()
    except Exception as e:
        print(f"[Agent 2] Error loading URLs: {e}")
        return f"Error: Failed to load one or more URLs. {e}"
        
    if not docs:
        return "Error: No documents were loaded from the provided URLs."

    # --- SPLIT ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    
    print(f"[Agent 2] Loaded {len(docs)} docs, split into {len(chunks)} chunks.")

    # Prepare data for insertion
    chunks_to_insert = []
    
    # Generate one document_id for all chunks from this scrape/doc batch
    # (A better approach might be one UUID per URL)
    
    # Let's do one UUID per source document
    doc_id_map = {doc.metadata.get('source'): uuid.uuid4() for doc in docs}

    for chunk in chunks:
        text_content = chunk.page_content
        source_url = chunk.metadata.get('source', 'N/A')
        document_id = doc_id_map.get(source_url)
        
        # Extract keywords
        keywords_tuples = kw_extractor.extract_keywords(text_content)
        extracted_keywords = [kw[0].lower() for kw in keywords_tuples]

        # Prepare entry
        chunks_to_insert.append({
            "chunk_id": uuid.uuid4(),
            "document_id": document_id,
            "source_url": source_url,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "text_content": text_content,
            "keywords": extracted_keywords,
        })
        
    # --- EMBED & STORE ---
    # Batch generate embeddings for efficiency
    all_text_content = [c["text_content"] for c in chunks_to_insert]
    
    if not all_text_content:
        return "No text content found in chunks to process."
        
    print(f"[Agent 2] Generating {len(all_text_content)} embeddings...")
    embeddings = embeddings_model.embed_documents(all_text_content)
    
    # Add embeddings to our data
    for i, chunk_data in enumerate(chunks_to_insert):
        chunk_data["embedding_vector"] = embeddings[i]

    # Batch insert into PostgreSQL
    try:
        with db_engine.connect() as conn:
            # We must use text() for the INSERT query
            insert_query = text(f"""
            INSERT INTO {TABLE_NAME} (
                chunk_id, document_id, source_url, timestamp, 
                text_content, keywords, embedding_vector
            ) VALUES (
                :chunk_id, :document_id, :source_url, :timestamp, 
                :text_content, :keywords, :embedding_vector
            )
            """)
            
            # Execute the batch insertion
            conn.execute(insert_query, chunks_to_insert)
            conn.commit()
            
    except Exception as e:
        error_msg = f"Error: Database insertion failed. {e}"
        print(f"[Agent 2] {error_msg}")
        return error_msg

    success_message = f"Successfully scraped, processed, and ingested content. Added {len(chunks)} new chunks to the '{TABLE_NAME}' table."
    print(f"[Agent 2] {success_message}")
    return success_message


@mcp.tool()
def semantic_search_tool(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Tool 2: Performs a semantic (vector) similarity search against the
    PostgreSQL 'embedding_vector' column.
    """
    print(f"[Agent 2] Performing semantic search for: '{query}'")
    
    # 1. Generate the query embedding
    try:
        query_embedding = embeddings_model.embed_query(query)
    except Exception as e:
        return [{"error": f"Failed to embed query: {e}"}]

    # 2. Execute the vector search query
    # We use the <-> (L2 distance) operator from pgvector
    search_query = text(f"""
    SELECT 
        text_content, 
        source_url, 
        (embedding_vector <-> :query_embedding) AS similarity
    FROM {TABLE_NAME}
    ORDER BY similarity
    LIMIT :k
    """)
    
    try:
        with db_engine.connect() as conn:
            results = conn.execute(search_query, {
                "query_embedding": query_embedding,
                "k": k
            })
            
            formatted_results = [
                {
                    "text_content": row.text_content,
                    "source": row.source_url,
                    "similarity_score": row.similarity
                } for row in results.mappings() # Use .mappings() to get dict-like rows
            ]
            
        print(f"[Agent 2] Found {len(formatted_results)} semantic results.")
        return formatted_results
        
    except Exception as e:
        error_msg = f"Error during semantic search: {e}"
        print(f"[Agent 2] {error_msg}")
        return [{"error": error_msg}]


@mcp.tool()
def keyword_search_tool(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Tool 3: Performs a keyword search against the 'keywords' TEXT[]
    array column in the PostgreSQL database.
    """
    print(f"[Agent 2] Performing keyword search for: '{query}'")
    
    # Split query into unique keywords
    query_keywords = list(set(query.lower().split()))
    
    # 2. Execute the array overlap query
    # We use the && (overlap) operator for PostgreSQL arrays
    search_query = text(f"""
    SELECT 
        text_content, 
        source_url, 
        keywords
    FROM {TABLE_NAME}
    WHERE keywords && :query_keywords
    LIMIT :limit
    """)
    
    try:
        with db_engine.connect() as conn:
            results = conn.execute(search_query, {
                "query_keywords": query_keywords,
                "limit": limit
            })
            
            formatted_results = [
                {
                    "text_content": row.text_content,
                    "source": row.source_url,
                    "matching_keywords": [kw for kw in row.keywords if kw in query_keywords]
                } for row in results.mappings()
            ]

        print(f"[Agent 2] Found {len(formatted_results)} keyword results.")
        return formatted_results

    except Exception as e:
        error_msg = f"Error during keyword search: {e}"
        print(f"[Agent 2] {error_msg}")
        return [{"error": error_msg}]


# --- 4. RUN THE MCP SERVER ---

if __name__ == "__main__":
    print("Setting up database...")
    setup_database()
    
    print("Starting Agent 2 (RAG & Web) MCP Tool Server...")
    # Run on port 8001 (or any port not used by other agents)
    mcp.run(transport="streamable-http", port=8001)
