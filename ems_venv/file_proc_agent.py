import os
from typing import Dict, Any
from multi_agent_docs.tools.file_proc_tools import (
    extract_code_elements,
    extract_dependencies,
    extract_file_relationships,
    suggest_skip_patterns,
)
from multi_agent_docs.tools.repo_intel_tools import read_file_content
from multi_agent_docs.services.llm_service import LLMService
import asyncio
import logging

logger = logging.getLogger(__name__)

MAX_FILES_TO_PROCESS_DEFAULT = 50

class FileProcessingAgent:
    def __init__(self, llm_api_key: str = None):
        self.llm = LLMService(api_key=llm_api_key)

    async def process(self, state: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        config = config or {}
        logger.info("Running File Processing Agent")
        project_path = state.get("project_path")
        repo_files = state.get("repo_files", [])

        if not project_path or not repo_files or state.get("repo_intel_error"):
            state["file_proc_error"] = "Project path or repo files missing or previous error."
            return state

        # Handle include/exclude config
        include_files = config.get("include_files")
        exclude_files = config.get("exclude_files", [])

        # Filter files by include/exclude lists
        filtered_files = repo_files
        if include_files:
            filtered_files = [f for f in repo_files if f in include_files]
        if exclude_files:
            filtered_files = [f for f in filtered_files if f not in exclude_files]

        max_files = config.get("max_files_to_process", MAX_FILES_TO_PROCESS_DEFAULT)
        files_to_process = filtered_files[:max_files]

        file_contents = {}
        for rel_path in files_to_process:
            content = read_file_content(project_path, rel_path)
            file_contents[rel_path] = content

        element_tasks = []
        dependency_tasks = []
        valid_files = []

        for rel_path, content in file_contents.items():
            if not content.startswith("Error"):
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
            state["code_elements"] = [{"error": str(e)}]
            state["dependencies"] = [{"error": str(e)}]
            state["file_relationships"] = [{"error": str(e)}]
            state["suggested_skip_patterns"] = [f"Error: {e}"]
            return state

        all_code_elements = [el for sublist in code_elements_results for el in sublist if not el.get("error")]
        valid_dependencies = [dep for dep in dependencies_results if not dep.get("error")]
        valid_relationships = [rel for rel in file_relationships if not rel.get("error")]

        state["file_contents"] = file_contents
        state["code_elements"] = all_code_elements
        state["dependencies"] = valid_dependencies
        state["file_relationships"] = valid_relationships
        state["suggested_skip_patterns"] = skip_patterns
        state["file_proc_error"] = None

        return state
