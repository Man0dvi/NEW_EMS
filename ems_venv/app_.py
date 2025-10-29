from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from multi_agent_docs.services.llm_service import LLMService

router = APIRouter()
llm = LLMService(api_key="YOUR_OPENAI_API_KEY")  # Init once, or inject

class SearchRequest(BaseModel):
    query: str
    final_state: dict  # Passed from the graph runner or client, includes analysis text

class SearchResponse(BaseModel):
    answer: str

@router.post("/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest):
    # Extract and flatten relevant textual context from final_state
    context_texts = []
    for key in ["tech_stack", "structure_summary", "architecture_summary", "complexity_assessment", "semantic_chunks"]:
        val = req.final_state.get(key)
        if val:
            if isinstance(val, list):
                context_texts.append("
".join(map(str, val)))
            else:
                context_texts.append(str(val))

    context = "

".join(context_texts)

    # Compose prompt with context + user query
    prompt = (
        f"You are a knowledgeable assistant. Use the following project analysis context to answer the question.

"
        f"{context}

"
        f"Question: {req.query}
Answer:"
    )

    try:
        answer = await llm.aask(prompt)  # or use your completion method
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM query failed: {e}")

    return SearchResponse(answer=answer)ere are the fully refactored key agent files and a simplified semantic search usage pattern according to your request: eliminate the separate semantic_search service, and use a shared `PGVector` vector store instance with `OpenAIEmbeddings` directly in each file.

***

### Shared vector store initialization (put in a shared config file or copy at top of each file)

```python
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

connection_string = "postgresql+psycopg2://langchain:langchain@localhost:5432/langchain"
collection_name = "my_docs"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key="YOUR_OPENAI_API_KEY"  # Set your API key accordingly
)

vector_store = PGVector(
    connection=connection_string,
    embeddings=embeddings,
    collection_name=collection_name,
    use_jsonb=True,
)
```

***

### Updated `code_discovery_agent.py`

```python
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from multi_agent_docs.services.llm_service import LLMService
from multi_agent_docs.services.parse_response import extract_json_from_llm_response

MAX_CONTENT_FOR_PROMPT_CHUNK = 80000
MAX_CHUNKS_FOR_PROMPT = 50
MAX_CONTENT_PREVIEW = 300

logger = logging.getLogger(__name__)

# Import vector store initialization here or from shared config
from your_vector_store_config import embeddings, vector_store  # Adapt import accordingly

async def chunk_code_semantically(filename: str, file_content: str, llm: LLMService) -> List[Dict[str, Any]]:
    # your existing chunking logic unchanged
    pass

async def enrich_chunks_with_metadata(chunks: List[Dict[str, Any]], llm: LLMService) -> List[Dict[str, Any]]:
    # your existing enrichment logic unchanged
    pass

class CodeDiscoveryAgent:
    def __init__(self, llm_api_key: str = None):
        self.llm = LLMService(api_key=llm_api_key)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Running Code Discovery Agent")
        project_path = state.get("project_path")
        file_contents = state.get("file_contents", {})

        if not project_path or not file_contents:
            state['code_discovery_error'] = "Missing project path or file contents."
            return state

        all_chunks = []
        for filename, content in file_contents.items():
            chunks = await chunk_code_semantically(filename, content, self.llm)
            enriched_chunks = await enrich_chunks_with_metadata(chunks, self.llm)
            all_chunks.extend(enriched_chunks)

        # Convert enriched chunks to Document objects and add them
        clean_documents = [
            Document(
                page_content=chunk.get("content", "").replace('\x00', ''),
                metadata={
                    "file_path": chunk.get("file", ""),
                    "start_line": chunk.get("start_line", 0),
                    "end_line": chunk.get("end_line", 0),
                    "code_role": chunk.get("code_role"),
                    "dependencies": chunk.get("dependencies"),
                    "business_context": chunk.get("business_context"),
                }
            )
            for chunk in all_chunks
        ]
        vector_store.add_documents(clean_documents)
        logger.info(f"Added {len(clean_documents)} code chunks to vector store.")

        state["semantic_chunks"] = all_chunks
        state["code_discovery_error"] = None
        return state
```

***

### Updated `file_proc_agent.py`

```python
import logging
from typing import Dict, Any
from langchain_core.documents import Document
from multi_agent_docs.services.llm_service import LLMService
from multi_agent_docs.tools.file_proc_tools import (
    extract_code_elements,
    extract_dependencies,
    extract_file_relationships,
    suggest_skip_patterns,
)
from multi_agent_docs.tools.repo_intel_tools import read_file_content
import asyncio

logger = logging.getLogger(__name__)

from your_vector_store_config import embeddings, vector_store  # Adapt import accordingly

MAX_FILES_TO_PROCESS = 50

class FileProcessingAgent:
    def __init__(self, llm_api_key: str = None):
        self.llm = LLMService(api_key=llm_api_key)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Running File Processing Agent")
        project_path = state.get("project_path")
        repo_files = state.get("repo_files", [])

        if not project_path or not repo_files or state.get("repo_intel_error"):
            state["file_proc_error"] = "Project path or repo files missing or previous error."
            return state

        files_to_process = repo_files[:MAX_FILES_TO_PROCESS]
        file_contents = {}
        for rel_path in files_to_process:
            content = read_file_content(project_path, rel_path)
            file_contents[rel_path] = content

        element_tasks = []
        dependency_tasks = []
        valid_files = []
        for rel_path, content in file_contents.items():
            if not content.startswith("Error:"):
                element_tasks.append(extract_code_elements(rel_path, content, self.llm))
                dependency_tasks.append(extract_dependencies(rel_path, content, self.llm))
                valid_files.append(rel_path)

        relationship_task = extract_file_relationships(valid_files, file_contents, self.llm) if len(valid_files) >= 2 else asyncio.sleep(0, result=[])
        skip_pattern_task = suggest_skip_patterns(repo_files, self.llm)

        try:
            code_elements_results = await asyncio.gather(*element_tasks)
            dependencies_results = await asyncio.gather(*dependency_tasks)
            file_relationships = await relationship_task
            skip_patterns = await skip_pattern_task
        except Exception as e:
            logger.error(f"Error in async tools: {e}")
            state["file_proc_error"] = str(e)
            # Set error placeholders
            state["code_elements"] = [{"error": str(e)}]
            state["dependencies"] = [{"error": str(e)}]
            state["file_relationships"] = [{"error": str(e)}]
            state["suggested_skip_patterns"] = [f"Error: {e}"]
            return state

        all_code_elements = [el for sublist in code_elements_results for el in sublist if not el.get("error")]

        # Add to vectorstore documents
        clean_documents = [
            Document(
                page_content=elem.get("code") or elem.get("content") or "",
                metadata={
                    "file_path": elem.get("file_path", ""),
                    "line_start": elem.get("line_start", 0),
                    "line_end": elem.get("line_end", 0),
                }
            )
            for elem in all_code_elements
        ]
        vector_store.add_documents(clean_documents)
        logger.info(f"Added {len(clean_documents)} code elements to vector store.")

        state["file_contents"] = file_contents
        state["code_elements"] = all_code_elements
        state["dependencies"] = [dep for dep in dependencies_results if not dep.get("error")]
        state["file_relationships"] = [rel for rel in file_relationships if not rel.get("error")]
        state["suggested_skip_patterns"] = skip_patterns
        state["file_proc_error"] = None

        return state
```

***

### Updated `repo_intel_agent.py`

```python
import os
import json
import logging
from typing import Dict, Any
from langchain_core.documents import Document
from multi_agent_docs.tools.repo_intel_tools import (
    list_project_files,
    detect_tech_stack,
    summarize_structure,
    compute_architecture,
    compute_complexity,
)
from multi_agent_docs.services.llm_service import LLMService

logger = logging.getLogger(__name__)

from your_vector_store_config import embeddings, vector_store  # Adapt import accordingly

class RepoIntelligenceAgent:
    def __init__(self, llm_api_key: str = None):
        self.llm = LLMService(api_key=llm_api_key)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Running Repo Intelligence Agent")
        project_path = state.get("project_path")
        if not project_path or not os.path.isdir(project_path):
            state["repo_intel_error"] = "Project path missing or invalid."
            return state

        repo_files = list_project_files(project_path)
        if not repo_files:
            state["repo_intel_error"] = "No relevant files found."
            state["repo_files"] = []
            return state
        state["repo_files"] = repo_files

        try:
            tech_stack = await detect_tech_stack(repo_files, self.llm)
            structure = await summarize_structure(repo_files, self.llm)
            architecture = await compute_architecture(repo_files, self.llm)
            complexity = await compute_complexity(repo_files, self.llm)
        except Exception as e:
            logger.error(f"Error during repo intel: {e}")
            state["repo_intel_error"] = str(e)
            return state

        state["tech_stack"] = tech_stack
        state["structure_summary"] = structure
        state["architecture_summary"] = architecture
        state["complexity_assessment"] = complexity
        state["repo_intel_error"] = None

        # Save repo structure summary embedding for semantic search
        if structure:
            summary_text = json.dumps(structure) if isinstance(structure, dict) else str(structure)
            doc = Document(page_content=summary_text, metadata={"type": "repo_summary"})
            vector_store.add_documents([doc])
            logger.info("Added repo structure summary embedding")

        return state
```

***

### Example FastAPI semantic search endpoint (use same vector_store instance):

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class SearchQuery(BaseModel):
    query: str
    top_k: int = 10

class SearchResultItem(BaseModel):
    content: str
    metadata: dict

class SearchResponse(BaseModel):
    results: list[SearchResultItem]

@router.post("/search/", response_model=SearchResponse)
async def search_code_snippets(search_query: SearchQuery):
    try:
        results = vector_store.similarity_search(query=search_query.query, k=search_query.top_k)
        items = [SearchResultItem(content=doc.page_content, metadata=doc.metadata) for doc in results]
        return SearchResponse(results=items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

***

This approach is:

- **Simplified:** No separate semantic_search.py or service abstraction.
- **Consistent:** One shared vector_store object is used everywhere.
- **Reliable:** Documents and vectors stored explicitly and cleaned.
- **Easy to maintain and debug.**

***

If you want, I can generate a single file with all these chunks integrated or help you with wiring the vector store in your existing FastAPI service. Just ask!
