# --- agents/file_proc_agent.py ---

import os
from typing import Dict, Any, List
from tools.file_proc_tools import (
    extract_code_elements,
    extract_dependencies,
    extract_file_relationships,
    suggest_skip_patterns,
)
# Need the tool to read file content
from tools.repo_intel_tools import read_file_content
from services.llm_service import LLMService
import asyncio # For running tools concurrently

# Limit how many files to process in detail to avoid excessive cost/time
MAX_FILES_TO_PROCESS = 50

class FileProcessingAgent:
    """ Agent for file-level analysis: extracts elements, dependencies, relationships. """

    def __init__(self, llm_api_key: str | None = None):
        self.llm = LLMService(api_key=llm_api_key)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """ Runs the file processing tools on relevant files. """
        print("\n--- Running File Processing Agent ---")
        project_path = state.get("project_path")
        repo_files = state.get("repo_files", [])

        if not project_path or not repo_files:
            print("Error: project_path or repo_files missing from state.")
            state["file_proc_error"] = "Project path or file list missing."
            return state

        # Select a subset of files for detailed processing
        files_to_process = repo_files[:MAX_FILES_TO_PROCESS]
        if len(repo_files) > MAX_FILES_TO_PROCESS:
            print(f"Warning: Processing details for first {MAX_FILES_TO_PROCESS} files only.")

        # --- Read file contents ---
        file_contents: Dict[str, str] = {}
        print(f"Reading content for {len(files_to_process)} files...")
        for rel_path in files_to_process:
            file_contents[rel_path] = read_file_content(project_path, rel_path) # Sync read

        # --- Run tools concurrently where possible ---
        print("Extracting code elements, dependencies...")
        element_tasks = []
        dependency_tasks = []
        for rel_path in files_to_process:
            content = file_contents.get(rel_path, "")
            if content and not content.startswith("Error:"):
                element_tasks.append(extract_code_elements(rel_path, content, self.llm))
                dependency_tasks.append(extract_dependencies(rel_path, content, self.llm))

        # Run relationship analysis (needs multiple files)
        relationship_task = extract_file_relationships(files_to_process, file_contents, self.llm)

        # Run skip pattern suggestion
        skip_pattern_task = suggest_skip_patterns(repo_files, self.llm) # Use full list here

        # Gather results
        code_elements_results = await asyncio.gather(*element_tasks)
        dependencies_results = await asyncio.gather(*dependency_tasks)
        file_relationships = await relationship_task
        skip_patterns = await skip_pattern_task

        # Flatten list of lists for elements
        all_code_elements = [element for sublist in code_elements_results for element in sublist if not sublist[0].get("error")]

        # Update state
        state["file_contents"] = file_contents # Store contents if needed downstream
        state["code_elements"] = all_code_elements
        state["dependencies"] = dependencies_results # List of dicts per file
        state["file_relationships"] = file_relationships
        state["suggested_skip_patterns"] = skip_patterns

        print("--- Finished File Processing Agent ---")
        return state
