# --- tools/repo_intel_tools.py ---

from typing import List, Dict, Any
from services.llm_service import LLMService # Assumes llm_service.py is in ../services/
import os
import json

# --- Constants ---
EXCLUDE_DIRS = {".git", ".vscode", "node_modules", "__pycache__", "dist", "build", "venv", ".venv"}
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024 # Skip files larger than 1MB
MAX_FILES_FOR_PROMPT = 100 # Limit number of files sent in prompts

# --- Helper Functions ---
def is_binary_file(filepath: str) -> bool:
    """ Basic check if a file appears to be binary. """
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(1024)
            # Simple check: presence of null bytes often indicates binary
            if b'\x00' in chunk:
                return True
            # Try decoding a small chunk, if it fails, likely binary
            chunk.decode('utf-8', errors='strict')
        return False
    except (IOError, UnicodeDecodeError):
        return True
    except Exception: # Catch other potential errors
        return True # Assume binary on unexpected error

def list_project_files(project_path: str) -> List[str]:
    """ Recursively collects non-binary, non-excluded files under a size limit. """
    print(f"[Repo Intel Tool] Listing files in: {project_path}")
    result = []
    if not os.path.isdir(project_path):
        print(f"Error: Provided path '{project_path}' is not a valid directory.")
        return []

    for root, dirs, files in os.walk(project_path, topdown=True):
        # Modify dirs in-place to prune excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file in files:
            file_path = os.path.join(root, file)
            # Use relative path for cleaner prompts
            relative_path = os.path.relpath(file_path, project_path)

            # Skip excluded dirs/files and large/binary files
            if any(part in EXCLUDE_DIRS for part in relative_path.split(os.sep)):
                continue
            try:
                if os.path.getsize(file_path) > MAX_FILE_SIZE_BYTES:
                    print(f"Skipping large file: {relative_path}")
                    continue
                if is_binary_file(file_path):
                    # print(f"Skipping binary file: {relative_path}") # Can be noisy
                    continue
                result.append(relative_path)
            except OSError as e:
                print(f"Error accessing file {relative_path}: {e}") # Handle permission errors etc.

    print(f"[Repo Intel Tool] Found {len(result)} relevant files.")
    return result

# --- LLM-Powered Tools ---
async def detect_tech_stack(files: List[str], llm: LLMService) -> List[str]:
    """ Uses LLM to guess tech stack based on file list. """
    print("[Repo Intel Tool] Detecting tech stack...")
    # Limit file list size for prompt
    files_sample = files[:MAX_FILES_FOR_PROMPT]
    if len(files) > MAX_FILES_FOR_PROMPT:
        print(f"Warning: Truncating file list to {MAX_FILES_FOR_PROMPT} for tech stack detection.")

    prompt = (
        f"Based on this list of project file paths, identify the primary programming languages, frameworks, and technologies used. Provide a concise list.\n\n"
        f"File Paths:\n{json.dumps(files_sample, indent=2)}\n\n"
        f"Detected Stack (list of strings):"
    )
    response = await llm.aask(prompt)
    # Basic parsing attempt
    try:
        # Check if response looks like a list
        if response.strip().startswith('[') and response.strip().endswith(']'):
            import ast
            stack = ast.literal_eval(response.strip())
            if isinstance(stack, list): return stack
        # Otherwise, return response split by lines/commas as fallback
        return [s.strip() for s in response.replace(',', '\n').split('\n') if s.strip()]
    except Exception:
        print("Warning: Could not parse tech stack list, returning raw response.")
        return [response] # Return raw LLM output if parsing fails

async def summarize_structure(files: List[str], llm: LLMService) -> Dict[str, Any]:
    """ Uses LLM to summarize codebase structure. """
    print("[Repo Intel Tool] Summarizing structure...")
    files_sample = files[:MAX_FILES_FOR_PROMPT]
    if len(files) > MAX_FILES_FOR_PROMPT:
        print(f"Warning: Truncating file list to {MAX_FILES_FOR_PROMPT} for structure summary.")

    prompt = (
        f"Analyze this list of project file paths:\n{json.dumps(files_sample, indent=2)}\n\n"
        f"Identify and categorize the files. Provide a summary in JSON format with keys like "
        f"'main_entrypoints' (list), 'configuration_files' (list), 'documentation_files' (list), "
        f"'test_files' (list), 'core_logic_folders' (list), and 'other_notable_files' (list)."
        f"Value for each key should be a list of relevant file paths."
    )
    response = await llm.aask(prompt)
    try:
        # LLM should return JSON directly
        summary = json.loads(response)
        return summary
    except json.JSONDecodeError:
        print("Warning: Could not parse structure summary JSON, returning raw.")
        return {"llm_raw_response": response}
    except Exception as e:
         print(f"Error parsing structure summary: {e}")
         return {"error": str(e), "llm_raw_response": response}

async def compute_architecture(files: List[str], llm: LLMService) -> str:
    """ Uses LLM for a natural language architecture summary. """
    print("[Repo Intel Tool] Computing architecture summary...")
    files_sample = files[:MAX_FILES_FOR_PROMPT]
    if len(files) > MAX_FILES_FOR_PROMPT:
        print(f"Warning: Truncating file list to {MAX_FILES_FOR_PROMPT} for architecture summary.")

    prompt = (
        f"Based on this list of project files:\n{json.dumps(files_sample, indent=2)}\n\n"
        f"Describe the high-level software architecture. What kind of application is it (e.g., web service, library, CLI)? "
        f"What are the likely key components or modules and how might they interact? Provide a concise natural language summary."
    )
    response = await llm.aask(prompt)
    return response

async def compute_complexity(files: List[str], llm: LLMService) -> str:
    """ Uses LLM to estimate project complexity. """
    print("[Repo Intel Tool] Computing complexity estimate...")
    files_sample = files[:MAX_FILES_FOR_PROMPT]
    if len(files) > MAX_FILES_FOR_PROMPT:
        print(f"Warning: Truncating file list to {MAX_FILES_FOR_PROMPT} for complexity estimate.")

    prompt = (
        f"Estimate the overall complexity of a codebase represented by these files:\n{json.dumps(files_sample, indent=2)}\n\n"
        f"Consider factors like the number of files, variety of languages/technologies, potential interdependencies, and likely project size. "
        f"Provide a complexity rating (e.g., Low, Medium, High, Very High) and a brief justification."
    )
    response = await llm.aask(prompt)
    return response

# --- NEW Tool: Read File Content ---
def read_file_content(project_path: str, relative_filepath: str) -> str:
    """ Reads the content of a specific file, given the project root and relative path. """
    print(f"[Repo Intel Tool] Reading file: {relative_filepath}")
    full_path = os.path.join(project_path, relative_filepath)
    try:
        # Check size again just in case
        if os.path.getsize(full_path) > MAX_FILE_SIZE_BYTES:
            return f"Error: File is too large (> {MAX_FILE_SIZE_BYTES / 1024 / 1024}MB)."
        # Check binary again
        if is_binary_file(full_path):
             return "Error: Cannot read binary file content."

        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Optional: Truncate very long files if needed for subsequent LLM calls
            # MAX_CONTENT_LENGTH = 100000 # Example limit
            # if len(content) > MAX_CONTENT_LENGTH:
            #     content = content[:MAX_CONTENT_LENGTH] + "\n... [TRUNCATED]"
            return content
    except FileNotFoundError:
        return f"Error: File not found at '{full_path}'."
    except Exception as e:
        return f"Error reading file '{relative_filepath}': {e}"
