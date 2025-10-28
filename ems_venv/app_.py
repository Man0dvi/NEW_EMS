# --- agents/repo_intel_agent.py ---

import os
from typing import Dict, Any, List
from tools.repo_intel_tools import (
    list_project_files,
    detect_tech_stack,
    summarize_structure,
    compute_architecture,
    compute_complexity
)
from services.llm_service import LLMService

class RepoIntelligenceAgent:
    """ Agent for initial codebase intelligence: lists files, detects stack, summarizes structure/architecture. """

    def __init__(self, llm_api_key: str | None = None):
        # Allow passing key or rely on environment variable
        self.llm = LLMService(api_key=llm_api_key)

    # Change to async process to match tools
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """ Runs the repo intelligence tools. """
        print("\n--- Running Repo Intelligence Agent ---")
        project_path = state.get("project_path")
        if not project_path or not os.path.isdir(project_path):
            print("Error: project_path is missing or invalid in state.")
            # Return state with error? Or raise exception?
            # Let's add an error message to the state
            state["repo_intel_error"] = "Project path missing or invalid."
            return state # Return state to allow graph to potentially handle error

        # Run tools
        repo_files = list_project_files(project_path) # Sync tool
        tech_stack = await detect_tech_stack(repo_files, self.llm) # Async tool
        structure = await summarize_structure(repo_files, self.llm) # Async tool
        architecture = await compute_architecture(repo_files, self.llm) # Async tool
        complexity = await compute_complexity(repo_files, self.llm) # Async tool

        # Update the state dictionary directly
        state["repo_files"] = repo_files
        state["tech_stack"] = tech_stack
        state["structure_summary"] = structure # Renamed key for clarity
        state["architecture_summary"] = architecture # Renamed key
        state["complexity_assessment"] = complexity # Renamed key

        print("--- Finished Repo Intelligence Agent ---")
        # Return the updated state dictionary
        return state
