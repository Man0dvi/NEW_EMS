import os
import json
import logging
from typing import List, Dict, Any
from multi_agent_docs.services.llm_service import LLMService
from multi_agent_docs.services.parse_response import extract_json_from_llm_response

# Constants for exclusions, max sizes, etc.
EXCLUDE_DIRS = {
    ".git", ".svn", ".hg", ".vscode", "node_modules", "__pycache__",
    "dist", "build", "target", "out", "bin", "obj",
    "venv", ".venv", "env", ".env",
    "docs", "doc", "example", "examples", "test", "tests", ".github", ".gitlab",
    "coverage", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "site", "vendor", "Pods", "Generated"
}
EXCLUDE_FILES = {
    "package-lock.json", "yarn.lock", "Pipfile.lock", "poetry.lock",
    ".gitignore", ".dockerignore", ".eslintignore", ".prettierignore",
    "LICENSE", "Makefile"
}
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024  # 1MB
MAX_FILES_FOR_PROMPT = 150

logger = logging.getLogger(__name__)

def is_binary_file(filepath: str) -> bool:
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(1024)
            return b'\x00' in chunk
    except Exception as e:
        logger.warning(f"Cannot check if binary for {filepath}: {e}")
        return True

def list_project_files(project_path: str) -> List[str]:
    logger.info(f"Listing files in: {project_path}")
    result = []
    if not os.path.isdir(project_path):
        logger.error(f"Invalid directory: {project_path}")
        return result
    for root, dirs, files in os.walk(project_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]

        for file in files:
            if file in EXCLUDE_FILES or file.startswith('.'):
                continue
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, project_path).replace("\\", "/")
            if any(part in EXCLUDE_DIRS for part in relative_path.split('/')):
                continue
            try:
                if os.path.getsize(file_path) > MAX_FILE_SIZE_BYTES:
                    continue
                if is_binary_file(file_path):
                    continue
                result.append(relative_path)
            except Exception as e:
                logger.warning(f"Failed access to {relative_path}: {e}")
    logger.info(f"Found {len(result)} files.")
    return result

async def detect_tech_stack(files: List[str], llm: LLMService) -> List[str]:
    if not files:
        return ["No files found to analyze."]
    files_sample = files[:MAX_FILES_FOR_PROMPT]
    prompt = (
        f"Identify primary technologies from these project files:\n{json.dumps(files_sample, indent=2)}"
    )
    response = await llm.aask(prompt)
    try:
        if response.strip().startswith('['):
            import ast
            stack = ast.literal_eval(response.strip())
            if isinstance(stack, list):
                return stack
        return [line.strip() for line in response.split('\n') if line.strip()]
    except Exception:
        return [response]

async def summarize_structure(files: List[str], llm: LLMService) -> Dict[str, Any]:
    if not files:
        return {"error": "No files found to analyze."}
    files_sample = files[:MAX_FILES_FOR_PROMPT]
    prompt = (
        f"Categorize these files into main_entrypoints, configuration_files, documentation_files, "
        f"test_files, core_logic_folders, utility_folders, and data_related_files. Output JSON only.\n"
        f"{json.dumps(files_sample, indent=2)}"
    )
    response = await llm.aask(prompt)
    try:
        summary = extract_json_from_llm_response(response)
        keys = ['main_entrypoints', 'configuration_files', 'documentation_files',
                'test_files', 'core_logic_folders', 'utility_folders', 'data_related_files']
        for key in keys:
            if key not in summary or not isinstance(summary[key], list):
                summary[key] = []
        return summary
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}, Raw: {response}")
        return {"error": f"Invalid structure JSON: {e}", "llm_raw": response}

async def compute_architecture(files: List[str], llm: LLMService) -> str:
    if not files:
        return "No files found to analyze."
    files_sample = files[:MAX_FILES_FOR_PROMPT]
    prompt = (
        f"Provide a concise high-level software architecture summary of this project:\n"
        f"{json.dumps(files_sample, indent=2)}"
    )
    response = await llm.aask(prompt)
    return response

async def compute_complexity(files: List[str], llm: LLMService) -> str:
    if not files:
        return "Complexity: Unknown (no files)."
    files_sample = files[:MAX_FILES_FOR_PROMPT]
    prompt = (
        f"Estimate the complexity level of the following project files:\n"
        f"{json.dumps(files_sample, indent=2)}"
    )
    response = await llm.aask(prompt)
    return response
