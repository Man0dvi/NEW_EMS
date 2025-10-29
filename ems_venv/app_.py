# routes.py (additional imports)
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from semantic_search import SemanticSearch
from database.db import get_db  # Assuming you have db session dependency
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize SemanticSearch service with your PG vector DB connection string
# Adjust the connection string as per your config
semantic_search_service = SemanticSearch(pgvector_conn_str="postgresql://user:password@localhost/dbname")

# Request body model for search queries
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5  # optional, default top 5 results

# Response model for search hits
class SearchResultItem(BaseModel):
    content: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResultItem]

# Semantic search endpoint
@router.post("/search/", response_model=SearchResponse)
async def search_code_snippets(search_query: SearchQuery, db: Session = Depends(get_db)):
    """
    Search relevant code snippets based on user query.
    Returns top-k most similar code chunks.
    """
    try:
        logger.info(f"Received semantic search query: {search_query.query}")
        # Perform semantic search against vector store
        docs = semantic_search_service.semantic_search(search_query.query, top_k=search_query.top_k)
        # Prepare response
        results = []
        for doc in docs:
            results.append(SearchResultItem(content=doc.page_content, metadata=doc.metadata))
        return SearchResponse(results=results)
    except Exception as e:
        logger.error(f"Error in semantic search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Semantic search error: {e}")


Let's start with the code changes for the three agents to emit embeddings and store the metadata using LangChain’s vector collection.

***

### 1. Changes for `code_discovery_agent.py`

Add embedding generation and storage after semantic chunk creation:

```python
# code_discovery_agent.py (partial)

from langchain.docstore.document import Document
from semantic_search import SemanticSearch  # your semantic search service class

class CodeDiscoveryAgent:
    def __init__(self, llmapikey: str = None, semantic_search: SemanticSearch = None):
        self.llm = LLMService(apikey=llmapikey)
        self.semantic_search = semantic_search

    async def process(self, state: dict) -> dict:
        # ... your existing code for chunkcodesemantically, enrichchunkswithmetadata
        semantic_chunks = await chunkcodesemantically(...)
        enriched_chunks = await enrichchunkswithmetadata(semantic_chunks, self.llm)
        state["semantic_chunks"] = enriched_chunks

        # Emit embeddings for storage and search
        if self.semantic_search and enriched_chunks:
            documents = []
            for chunk in enriched_chunks:
                content = chunk.get("text") or chunk.get("content") or ""
                metadata = {
                    "file_path": chunk.get("file_path", ""),
                    "start_line": chunk.get("start_line", 0),
                    "end_line": chunk.get("end_line", 0),
                }
                documents.append(Document(page_content=content, metadata=metadata))
            self.semantic_search.vectorstore.add_documents(documents)
            # Optionally log
            print(f"Added {len(documents)} semantic chunks to vector store")

        return state
```

You need to create and pass a `SemanticSearch` instance initialized with your vector DB connection, e.g. in your API setup or routes.

***

### 2. Changes for `repo_intel_agent.py`

Add embedding after repo intelligence, if relevant:

```python
# repo_intel_agent.py (partial)

from langchain.docstore.document import Document
from semantic_search import SemanticSearch

class RepoIntelligenceAgent:
    def __init__(self, llmapikey: str = None, semantic_search: SemanticSearch = None):
        self.llm = LLMService(apikey=llmapikey)
        self.semantic_search = semantic_search

    async def process(self, state: dict) -> dict:
        # Your existing repo intelligence processing code here
        # e.g., file listing, tech detection, structure summary generation

        # Optionally generate embeddings for repo-level summary or important files
        # If you have meaningful textual data to embed, create documents and add them
        repo_summary = state.get("structure_summary", {}).get("summary_text")  # example
        if repo_summary and self.semantic_search:
            document = Document(page_content=repo_summary, metadata={"type": "repo_summary"})
            self.semantic_search.vectorstore.add_documents([document])
            print("Added repo summary embedding")

        return state
```

Pass `SemanticSearch` instance similarly.

***

### 3. Changes for `file_proc_agent.py`

Add embeddings for file chunks or extracted code elements:

```python
# file_proc_agent.py (partial)

from langchain.docstore.document import Document
from semantic_search import SemanticSearch

class FileProcessingAgent:
    def __init__(self, llmapikey: str = None, semantic_search: SemanticSearch = None):
        self.llm = LLMService(apikey=llmapikey)
        self.semantic_search = semantic_search

    async def process(self, state: dict) -> dict:
        # Your file processing logic (code elements extraction etc.)
        code_elements = state.get("code_elements", [])

        # Generate embeddings from code elements if semantic search is enabled
        if self.semantic_search and code_elements:
            documents = []
            for elem in code_elements:
                content = elem.get("code") or elem.get("content") or ""
                metadata = {
                    "file_path": elem.get("file_path", ""),
                    "line_start": elem.get("line_start", 0),
                    "line_end": elem.get("line_end", 0),
                }
                documents.append(Document(page_content=content, metadata=metadata))
            self.semantic_search.vectorstore.add_documents(documents)
            print(f"Added {len(documents)} file elements to vector store.")

        return state
```

***

### Integration Notes:

- Initialize `SemanticSearch` instance once per request or lifecycle with your vector DB connection.
- Pass the instance to agent constructors or to `process()` methods if preferred.
- Agents emit new embeddings to the vector store on their processed chunks.
- You can extend metadata to include user/project IDs for multi-tenant support.

***

Shall we proceed to code sample for integrating search endpoint in `routes.py` next?
