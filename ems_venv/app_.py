# --- agents/code_discovery_agent.py ---

import os
from typing import Dict, Any, List
from tools.code_discovery_tools import (
    chunk_code_semantically,
    enrich_chunks_with_metadata,
    generate_persona_insights,
)
# Need file reading tool if chunking happens here
from tools.repo_intel_tools import read_file_content
from services.llm_service import LLMService
import asyncio
import json # For handling chunk data

# Limit how many files to chunk/process
MAX_FILES_FOR_SEMANTIC_CHUNK = 20

class CodeDiscoveryAgent:
    """ Agent for semantic chunking, metadata enrichment, and persona insights. """

    def __init__(self, llm_api_key: str | None = None):
        self.llm = LLMService(api_key=llm_api_key)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """ Runs semantic chunking and analysis tools. """
        print("\n--- Running Code Discovery Agent ---")
        project_path = state.get("project_path")
        # Use files identified by repo intel, maybe filtered by file proc?
        repo_files = state.get("repo_files", [])
        personas = state.get("personas", ["SDE", "PM"]) # Get target personas from state

        if not project_path or not repo_files:
            print("Error: project_path or repo_files missing from state.")
            state["code_discovery_error"] = "Project path or file list missing."
            return state

        # --- Semantic Chunking ---
        # Select files to chunk (e.g., source code files identified earlier)
        # For simplicity, let's chunk a subset of the repo files
        files_to_chunk = repo_files[:MAX_FILES_FOR_SEMANTIC_CHUNK]
        if len(repo_files) > MAX_FILES_FOR_SEMANTIC_CHUNK:
             print(f"Warning: Semantically chunking first {MAX_FILES_FOR_SEMANTIC_CHUNK} files only.")

        print(f"Starting semantic chunking for {len(files_to_chunk)} files...")
        chunking_tasks = []
        for rel_path in files_to_chunk:
            # Read content specifically for chunking
            content = read_file_content(project_path, rel_path)
            if content and not content.startswith("Error:"):
                 chunking_tasks.append(chunk_code_semantically(rel_path, content, self.llm))
            else:
                 print(f"Skipping chunking for {rel_path} due to read error or binary content.")

        # Gather all chunks
        chunk_results = await asyncio.gather(*chunking_tasks)
        all_chunks = [chunk for sublist in chunk_results for chunk in sublist if not sublist[0].get("error")]
        print(f"Generated {len(all_chunks)} semantic chunks.")

        if not all_chunks:
             state["semantic_chunks"] = []
             state["persona_insights"] = {}
             print("No semantic chunks generated.")
             print("--- Finished Code Discovery Agent ---")
             return state

        # --- Enrich Chunks ---
        print("Enriching chunks with metadata...")
        enriched_chunks = await enrich_chunks_with_metadata(all_chunks, self.llm)

        # --- Generate Persona Insights ---
        print(f"Generating insights for personas: {personas}...")
        # Start with enriched chunks
        chunks_for_persona = enriched_chunks
        persona_outputs = {}
        # Apply insights sequentially or concurrently if desired
        for persona in personas:
            chunks_with_persona_insight = await generate_persona_insights(chunks_for_persona, self.llm, persona)
            # Update the main list of chunks for the next persona (optional)
            # chunks_for_persona = chunks_with_persona_insight
            # Store final result keyed by persona
            persona_outputs[persona] = chunks_with_persona_insight # Store the chunks with insights

        # Update state with final results
        # Store the enriched chunks *with* the final persona insights added
        final_enriched_chunks_with_insights = chunks_with_persona_insight if personas else enriched_chunks

        state["semantic_chunks"] = final_enriched_chunks_with_insights
        # Optional: Keep persona_outputs separate if needed
        # state["persona_insights"] = persona_outputs

        print("--- Finished Code Discovery Agent ---")
        return state
