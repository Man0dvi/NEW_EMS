import os
import logging
from typing import Dict, Any
from multi_agent_docs.services.llm_service import LLMService
from multi_agent_docs.tools.repo_intel_tools import (
    list_project_files,
    detect_tech_stack,
    summarize_structure,
    compute_architecture,
    compute_complexity,
)

logger = logging.getLogger(__name__)

class RepoIntelligenceAgent:
    def __init__(self, llm_api_key: str = None):
        self.llm = LLMService(api_key=llm_api_key)

    async def process(self, state: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        logger.info("Running Repo Intelligence Agent")
        project_path = state.get("project_path")
        if not project_path or not os.path.isdir(project_path):
            state["repo_intel_error"] = "Invalid or missing project path."
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

        return state
