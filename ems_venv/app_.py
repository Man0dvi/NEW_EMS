from typing import List, Dict, Any
from langchain.docstore.document import Document
from multi_agent_docs.services.llm_service import LLMService
from multi_agent_docs.services.parse_response import extract_json_from_llm_response
from semantic_search import SemanticSearch  # Your semantic search service
import logging
import asyncio

# --- Constants ---
MAX_CONTENT_FOR_PROMPT_CHUNK = 80000
MAX_CHUNKS_FOR_PROMPT = 50
MAX_CONTENT_PREVIEW = 300

logger = logging.getLogger(__name__)

# --- LLM-Powered Tools ---
async def chunk_code_semantically(filename: str, file_content: str, llm: LLMService) -> List[Dict[str, Any]]:
    logger.info(f"Semantically chunking: {filename}")
    if not file_content or file_content.startswith("Error:"):
        return [{"file": filename, "error": "Invalid or missing content for chunking."}]
    content_to_chunk = file_content
    if len(file_content) > MAX_CONTENT_FOR_PROMPT_CHUNK:
        logger.warning(f"Content for {filename} truncated for semantic chunking.")
        content_to_chunk = file_content[:MAX_CONTENT_FOR_PROMPT_CHUNK] + "\n... [TRUNCATED]"

    prompt = (
        f"Analyze source code from '{filename}':\n``````\n\n"
        f"Divide code into meaningful semantic chunks (functions, classes, methods, config blocks, standalone scripts). "
        f"Return ONLY a valid JSON list of objects. Each object must have keys:\n"
        f"- 'type': string (e.g., 'function', 'class', 'method', 'config_block')\n"
        f"- 'name': string (e.g., function/class name or 'ConfigBlock1')\n"
        f"- 'start_line': int (approximate)\n"
        f"- 'end_line': int (approximate)\n"
        f"- 'content': string (the actual code chunk)\n"
        f"- 'summary': string (1 concise sentence of purpose)"
    )
    response = await llm.aask(prompt)
    try:
        chunks = extract_json_from_llm_response(response)
        if not isinstance(chunks, list): raise ValueError("Expected a list")
        validated = []
        for i, ch in enumerate(chunks):
            if isinstance(ch, dict) and all(k in ch for k in ['type', 'name', 'start_line', 'end_line', 'content', 'summary']):
                ch['file'] = filename
                ch['chunk_id'] = f"{filename}_{i}"  # Add unique ID
                validated.append(ch)
            else:
                logger.warning(f"Skipping invalid chunk structure in {filename}: {ch}")
        return validated
    except Exception as e:
        logger.error(f"Error parsing semantic chunks JSON for {filename}: {e}. Raw: {response}")
        return [{"file": filename, "chunk_id": f"{filename}_0", "type": "file", "name": filename,
                 "start_line": 1, "end_line": file_content.count('\n') + 1, "content": file_content,
                 "summary": "Error during chunking.", "error": str(e), "llm_raw_response": response}]

async def enrich_chunks_with_metadata(chunks: List[Dict[str, Any]], llm: LLMService) -> List[Dict[str, Any]]:
    logger.info(f"Enriching {len(chunks)} chunks with metadata...")
    if not chunks: return []

    chunks_batch = chunks[:MAX_CHUNKS_FOR_PROMPT]
    if len(chunks) > MAX_CHUNKS_FOR_PROMPT:
        logger.warning(f"Processing only first {MAX_CHUNKS_FOR_PROMPT} chunks for enrichment.")

    prompt_chunks = []
    for i, chunk in enumerate(chunks_batch):
        prompt_chunks.append({
            "id": chunk.get('chunk_id', f"chunk_{i}"),
            "file": chunk.get('file', '?'), "type": chunk.get('type', '?'), "name": chunk.get('name', '?'),
            "summary": chunk.get('summary', ''),
            "content_preview": chunk.get('content', '')[:MAX_CONTENT_PREVIEW] + "..."
        })

    prompt = (
        f"Analyze code chunks:\n{json.dumps(prompt_chunks, indent=2)}\n\n"
        f"For each chunk (by 'id'), determine 'code_role' (e.g., 'API Endpoint', 'Business Logic', 'Data Model', 'Utility', 'Config', 'Test'), "
        f"main 'dependencies' (libs/modules from preview/summary), and brief 'business_context' (1 sentence relevance/risk).\n"
        f"Return ONLY a valid JSON list. Each object must contain 'id' and the new keys: 'code_role'(string), 'dependencies'(list[string]), 'business_context'(string)."
    )
    response = await llm.aask(prompt)
    try:
        enrichment_data = extract_json_from_llm_response(response)
        if not isinstance(enrichment_data, list): raise ValueError("Expected list")
        enrichment_map = {item.get('id'): item for item in enrichment_data if isinstance(item, dict) and 'id' in item}

        for chunk in chunks_batch:
            chunk_id = chunk.get('chunk_id')
            enrichment = enrichment_map.get(chunk_id) if chunk_id else None
            if enrichment:
                chunk['code_role'] = enrichment.get('code_role', 'Unknown')
                chunk['dependencies'] = enrichment.get('dependencies', [])
                chunk['business_context'] = enrichment.get('business_context', 'N/A')
            else:
                chunk['code_role'] = 'Error'
                chunk['dependencies'] = []
                chunk['business_context'] = 'Enrichment failed.'

        return chunks_batch + chunks[MAX_CHUNKS_FOR_PROMPT:]
    except Exception as e:
        logger.error(f"Error parsing enrichment JSON: {e}. Raw: {response}")
        for chunk in chunks_batch:
            chunk['code_role'] = 'Error'
            chunk['dependencies'] = []
            chunk['business_context'] = f'Enrichment Error: {e}'
        return chunks

async def generate_persona_insights(chunks: List[Dict[str, Any]], llm: LLMService, persona: str) -> List[Dict[str, Any]]:
    logger.info(f"Generating insights for persona: {persona} on {len(chunks)} chunks...")
    if not chunks: return []

    chunks_batch = chunks[:MAX_CHUNKS_FOR_PROMPT]
    if len(chunks) > MAX_CHUNKS_FOR_PROMPT:
        logger.warning(f"Processing only first {MAX_CHUNKS_FOR_PROMPT} chunks for persona insights.")

    prompt_chunks = []
    for i, chunk in enumerate(chunks_batch):
        prompt_chunks.append({
            "id": chunk.get('chunk_id', f"chunk_{i}"),
            "file": chunk.get('file', '?'), "type": chunk.get('type', '?'), "name": chunk.get('name', '?'),
            "summary": chunk.get('summary', '?'), "code_role": chunk.get('code_role', '?'),
            "dependencies": chunk.get('dependencies', []), "business_context": chunk.get('business_context', '?'),
        })

    if persona.lower() == 'sde':
        persona_instr = "Focus on technical complexity, dependencies, testability, refactoring needs for a Software Dev Engineer."
    elif persona.lower() == 'pm':
        persona_instr = "Focus on business logic, user features, roadmap impact, risks for a Product Manager."
    else:
        persona_instr = "Provide a general overview."

    prompt = (
        f"Analyze enriched code chunks for persona '{persona}':\n{json.dumps(prompt_chunks, indent=2)}\n\n"
        f"For each chunk (by 'id'), provide a concise 'persona_insight' (1-2 sentences). {persona_instr}\n"
        f"Return ONLY a valid JSON list. Each object must contain 'id' and 'persona_insight'(string)."
    )
    response = await llm.aask(prompt)
    try:
        insight_data = extract_json_from_llm_response(response)
        if not isinstance(insight_data, list): raise ValueError("Expected list")
        insight_map = {item.get('id'): item.get('persona_insight', 'N/A') for item in insight_data if isinstance(item, dict) and 'id' in item}

        insight_key = f'insight_{persona.lower()}'
        for chunk in chunks_batch:
            chunk_id = chunk.get('chunk_id')
            chunk[insight_key] = insight_map.get(chunk_id, 'Failed to generate insight.')

        return chunks_batch + chunks[MAX_CHUNKS_FOR_PROMPT:]
    except Exception as e:
        logger.error(f"Error parsing persona insight JSON ({persona}): {e}. Raw: {response}")
        insight_key = f'insight_{persona.lower()}'
        for chunk in chunks_batch:
            chunk[insight_key] = f'Insight Error: {e}'
        return chunks


class CodeDiscoveryAgent:
    def __init__(self, llm_api_key: str = None, semantic_search: SemanticSearch = None, websocket_broadcaster=None):
        self.llm = LLMService(api_key=llm_api_key)
        self.semantic_search = semantic_search
        self.websocket_broadcaster = websocket_broadcaster

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Running Code Discovery Agent")

        project_path = state.get("project_path")
        file_contents = state.get("file_contents", {})

        if not project_path or not file_contents:
            state['code_discovery_error'] = "Missing project path or file contents."
            return state

        # Emit a websocket event: start chunking
        if self.websocket_broadcaster:
            await self.websocket_broadcaster.broadcast({
                "event": "chunking_started",
                "message": f"Starting to chunk {len(file_contents)} files."
            })

        # Chunk and enrich code files
        all_chunks = []
        for filename, content in file_contents.items():
            chunks = await chunk_code_semantically(filename, content, self.llm)
            enriched_chunks = await enrich_chunks_with_metadata(chunks, self.llm)
            all_chunks.extend(enriched_chunks)

            # Emit a websocket event per file chunked
            if self.websocket_broadcaster:
                await self.websocket_broadcaster.broadcast({
                    "event": "file_chunked",
                    "message": f"Chunked and enriched file: {filename}"
                })

        # Store chunks in vector store for RAG search
        if self.semantic_search and all_chunks:
            documents = []
            for chunk in all_chunks:
                content = chunk.get("content") or chunk.get("text") or ""
                metadata = {
                    "file_path": chunk.get("file", ""),
                    "start_line": chunk.get("start_line", 0),
                    "end_line": chunk.get("end_line", 0),
                }
                documents.append(Document(page_content=content, metadata=metadata))
            self.semantic_search.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} code chunks to vector store.")

            # Emit a websocket event: indexing completed
            if self.websocket_broadcaster:
                await self.websocket_broadcaster.broadcast({
                    "event": "indexing_completed",
                    "message": f"Indexed {len(documents)} code chunks for semantic search."
                })

        state["semantic_chunks"] = all_chunks
        state["code_discovery_error"] = None

        logger.info("Finished Code Discovery Agent")
        return state
