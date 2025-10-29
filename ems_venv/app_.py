import os
from typing import Dict, Any
from langchain_core.documents import Document
from multi_agent_docs.tools.file_proc_tools import (
    extract_code_elements,
    extract_dependencies,
    extract_file_relationships,
    suggest_skip_patterns,
)
from multi_agent_docs.tools.repo_intel_tools import read_file_content
from multi_agent_docs.services.llm_service import LLMService
from semantic_search import SemanticSearch
import asyncio
import logging

logger = logging.getLogger(__name__)
MAX_FILES_TO_PROCESS = 50

class FileProcessingAgent:
    def __init__(self, llm_api_key: str = None, semantic_search: SemanticSearch = None, websocket_broadcaster=None):
        self.llm = LLMService(api_key=llm_api_key)
        self.semantic_search = semantic_search
        self.websocket_broadcaster = websocket_broadcaster

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Running File Processing Agent")

        project_path = state.get("project_path")
        repo_files = state.get("repo_files", [])

        if not project_path or not repo_files:
            state["file_proc_error"] = "Project path or file list missing."
            return state
        if state.get("repo_intel_error"):
            state["file_proc_error"] = "Skipped due to repo intel error."
            return state

        files_to_process = repo_files[:MAX_FILES_TO_PROCESS]
        if len(repo_files) > MAX_FILES_TO_PROCESS:
            logger.warning(f"Processing details for first {MAX_FILES_TO_PROCESS} files.")

        if self.websocket_broadcaster:
            await self.websocket_broadcaster.broadcast({ "event": "file_reading_started", "message": f"Reading content for {len(files_to_process)} files." })

        file_contents = {}
        read_errors = 0
        for rel_path in files_to_process:
            content = read_file_content(project_path, rel_path)
            if content.startswith("Error:"):
                logger.warning(f"Read Error for {rel_path}: {content}")
                read_errors += 1
            file_contents[rel_path] = content

            if self.websocket_broadcaster:
                await self.websocket_broadcaster.broadcast({ "event": "file_read", "message": f"Read file {rel_path}" })

        if self.websocket_broadcaster:
            await self.websocket_broadcaster.broadcast({ "event": "file_reading_completed", "message": f"Finished reading files with {read_errors} errors." })

        element_tasks = []
        dependency_tasks = []
        valid_content_files = []
        for rel_path, content in file_contents.items():
            if not content.startswith("Error:"):
                element_tasks.append(extract_code_elements(rel_path, content, self.llm))
                dependency_tasks.append(extract_dependencies(rel_path, content, self.llm))
                valid_content_files.append(rel_path)

        if len(valid_content_files) >= 2:
            relationship_task = extract_file_relationships(valid_content_files, file_contents, self.llm)
        else:
            relationship_task = asyncio.sleep(0, result=[])
            logger.warning("Skipping relationship analysis: Not enough valid files.")

        skip_pattern_task = suggest_skip_patterns(repo_files, self.llm)

        try:
            logger.info("Gathering file processing results...")
            code_elements_results = await asyncio.gather(*element_tasks)
            dependencies_results = await asyncio.gather(*dependency_tasks)
            file_relationships = await relationship_task
            skip_patterns = await skip_pattern_task
            logger.info("Finished gathering results.")
        except Exception as e:
            logger.error(f"Error during async tool execution: {e}")
            state["file_proc_error"] = f"Async tool execution failed: {e}"
            state["code_elements"] = [{"error": str(e)}]
            state["dependencies"] = [{"error": str(e)}]
            state["file_relationships"] = [{"error": str(e)}]
            state["suggested_skip_patterns"] = [f"Error: {e}"]
            return state

        all_code_elements = [el for sublist in code_elements_results for el in sublist if not el.get("error")]
        valid_dependencies = [dep for dep in dependencies_results if not dep.get("error")]
        valid_relationships = [rel for rel in file_relationships if not rel.get("error")]

        if self.semantic_search and all_code_elements:
            documents = [
                Document(page_content=elem.get("code") or elem.get("content") or "",
                         metadata={
                             "file_path": elem.get("file_path", ""),
                             "line_start": elem.get("line_start", 0),
                             "line_end": elem.get("line_end", 0),
                         })
                for elem in all_code_elements
            ]
            self.semantic_search.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} code elements to vector store.")

            if self.websocket_broadcaster:
                await self.websocket_broadcaster.broadcast({
                    "event": "embedding_indexed",
                    "message": f"Indexed {len(documents)} code elements for semantic search."
                })

        state["file_contents"] = file_contents
        state["code_elements"] = all_code_elements
        state["dependencies"] = valid_dependencies
        state["file_relationships"] = valid_relationships
        state["suggested_skip_patterns"] = skip_patterns
        state["file_proc_error"] = None

        logger.info("Finished File Processing Agent")
        return state
