import os
import json
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
from semantic_search import SemanticSearch  # your semantic search wrapper
import logging

logger = logging.getLogger(__name__)

class RepoIntelligenceAgent:
    """Agent for initial codebase intelligence."""
    def __init__(self, llm_api_key: str | None = None,
                 semantic_search: SemanticSearch = None,
                 websocket_broadcaster=None):
        self.llm = LLMService(api_key=llm_api_key)
        self.semantic_search = semantic_search
        self.websocket_broadcaster = websocket_broadcaster

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Running Repo Intelligence Agent")

        project_path = state.get("project_path")
        if not project_path or not os.path.isdir(project_path):
            state["repo_intel_error"] = "Project path missing or invalid."
            return state

        repo_files = list_project_files(project_path)
        if not repo_files:
            state["repo_intel_error"] = "No relevant files found in the project path."
            state["repo_files"] = []
            return state
        state["repo_files"] = repo_files

        if self.websocket_broadcaster:
            await self.websocket_broadcaster.broadcast({
                "event": "repo_intel_started",
                "message": f"Starting repo intelligence on {len(repo_files)} files."
            })

        try:
            tech_stack = await detect_tech_stack(repo_files, self.llm)
            structure = await summarize_structure(repo_files, self.llm)
            architecture = await compute_architecture(repo_files, self.llm)
            complexity = await compute_complexity(repo_files, self.llm)
        except Exception as e:
            logger.error(f"Error during repo intel async tool execution: {e}")
            state["repo_intel_error"] = f"LLM analysis failed: {e}"
            # Still return partial results if available
            state["tech_stack"] = state.get("tech_stack", ["Error"])
            state["structure_summary"] = state.get("structure_summary", {"error": "LLM failed"})
            state["architecture_summary"] = state.get("architecture_summary", "Error")
            state["complexity_assessment"] = state.get("complexity_assessment", "Error")

            if self.websocket_broadcaster:
                await self.websocket_broadcaster.broadcast({
                    "event": "repo_intel_failed",
                    "message": f"Repo intelligence failed: {e}"
                })

            return state

        state["tech_stack"] = tech_stack
        state["structure_summary"] = structure
        state["architecture_summary"] = architecture
        state["complexity_assessment"] = complexity
        state["repo_intel_error"] = None

        # Broadcast completion
        if self.websocket_broadcaster:
            await self.websocket_broadcaster.broadcast({
                "event": "repo_intel_completed",
                "message": f"Repo intelligence complete."
            })

        # Add repo structure summary (or other metadata) to vector store for semantic search
        if self.semantic_search and structure:
            summary_text = json.dumps(structure) if isinstance(structure, dict) else str(structure)
            doc = Document(page_content=summary_text, metadata={"type": "repo_summary"})
            self.semantic_search.vectorstore.add_documents([doc])
            logger.info("Added repo structure summary embedding")

            if self.websocket_broadcaster:
                await self.websocket_broadcaster.broadcast({
                    "event": "embedding_indexed",
                    "message": "Indexed repo structure summary for semantic search."
                })

        logger.info("Finished Repo Intelligence Agent")
        return state
