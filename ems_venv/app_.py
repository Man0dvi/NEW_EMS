Okay, let's complete the code for your repository intelligence agent based on the structure you've provided and integrate it into your FastAPI application.

This involves:

1.  Completing the `tools` files (`repo_intel_tools.py`, `file_proc_tools.py`, `code_discovery_tools.py`).
2.  Completing the `agents` files (`repo_intel_agent.py`, `file_proc_agent.py`, `code_discovery_agent.py`).
3.  Completing the `graph` files (`nodes.py`, `graph.py`).
4.  Adding a new endpoint to `api/routes.py` to trigger the analysis graph.

-----

### 1\. `services/llm_service.py`

(Assuming this file exists as provided in the previous step, handling OpenAI initialization and `ask`/`aask` methods).

-----

### 2\. `tools/repo_intel_tools.py` (Completed)

```python
# --- tools/repo_intel_tools.py ---

from typing import List, Dict, Any
# Adjust import path based on your actual project structure
# Example assumes services/, tools/, agents/, graph/ are siblings
from services.llm_service import LLMService
import os
import json
import logging # Use logging for better output control

# --- Constants ---
EXCLUDE_DIRS = {
    ".git", ".svn", ".hg", ".vscode", "node_modules", "__pycache__",
    "dist", "build", "target", "out", "bin", "obj",
    "venv", ".venv", "env", ".env",
    "docs", "doc", "example", "examples", "test", "tests", ".github", ".gitlab",
    "coverage", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "site", "vendor", "Pods", "Generated" # Add more common ones
}
EXCLUDE_FILES = {
    "package-lock.json", "yarn.lock", "Pipfile.lock", "poetry.lock",
    ".gitignore", ".dockerignore", ".eslintignore", ".prettierignore",
    "LICENSE", "Makefile", # Often less relevant for code structure
    # Add common config files if summarized separately, e.g. Dockerfile, *.yml
}
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024 # 1MB limit
MAX_FILES_FOR_PROMPT = 150 # Limit for prompts needing file lists

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def is_binary_file(filepath: str) -> bool:
    """ Basic check if a file appears to be binary using null byte presence. """
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(1024)
            return b'\x00' in chunk
    except IOError as e:
        logger.warning(f"IOError checking if binary {filepath}: {e}")
        return True # Assume binary on read error
    except Exception as e:
        logger.error(f"Unexpected error checking binary {filepath}: {e}")
        return True # Assume binary on unexpected error

def list_project_files(project_path: str) -> List[str]:
    """ Recursively collects non-binary, non-excluded files under size limit. Returns relative paths. """
    logger.info(f"Listing files in: {project_path}")
    result = []
    if not os.path.isdir(project_path):
        logger.error(f"Provided path '{project_path}' is not a valid directory.")
        return []
    for root, dirs, files in os.walk(project_path, topdown=True):
        # Filter directories to prevent descending into excluded ones
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]

        for file in files:
            # Basic file name exclusions
            if file in EXCLUDE_FILES or file.startswith('.'): continue

            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, project_path).replace("\\", "/") # Normalize

            # Check if path contains excluded dir component (redundant check, belt-and-suspenders)
            if any(part in EXCLUDE_DIRS for part in relative_path.split('/')): continue

            try:
                # Check size and binary status
                if os.path.getsize(file_path) > MAX_FILE_SIZE_BYTES:
                    # logger.debug(f"Skipping large file: {relative_path}") # Debug level
                    continue
                if is_binary_file(file_path):
                    # logger.debug(f"Skipping binary file: {relative_path}") # Debug level
                    continue
                result.append(relative_path)
            except OSError as e:
                logger.warning(f"Cannot access file {relative_path}: {e}") # Handle permission errors

    logger.info(f"Found {len(result)} relevant files.")
    return result

# --- LLM-Powered Tools ---
async def detect_tech_stack(files: List[str], llm: LLMService) -> List[str]:
    """ Uses LLM to guess tech stack based on file list. """
    logger.info("Detecting tech stack...")
    if not files: return ["No files found to analyze."]
    files_sample = files[:MAX_FILES_FOR_PROMPT]
    prompt = (
        f"Based ONLY on the following file paths from a software project, identify the primary programming languages, frameworks, databases, and relevant technologies used. Be concise and list the key technologies.\n\n"
        f"File Paths:\n{json.dumps(files_sample, indent=2)}\n\n"
        f"Detected Stack (list of strings):"
    )
    response = await llm.aask(prompt)
    try: # Try parsing as list
        if response.strip().startswith('['):
            import ast; stack = ast.literal_eval(response.strip()); return stack if isinstance(stack, list) else [response]
        return [s.strip() for s in response.replace(',', '\n').split('\n') if s.strip() and len(s.strip()) > 1] # Fallback parsing
    except: return [response] # Raw response on error

async def summarize_structure(files: List[str], llm: LLMService) -> Dict[str, Any]:
    """ Uses LLM to summarize codebase structure into categories. """
    logger.info("Summarizing structure...")
    if not files: return {"error": "No files found to analyze."}
    files_sample = files[:MAX_FILES_FOR_PROMPT]
    prompt = (
        f"Analyze this list of relative file paths:\n{json.dumps(files_sample, indent=2)}\n\n"
        f"Categorize these files. Identify likely 'main_entrypoints', 'configuration_files', 'documentation_files' (like README, CONTRIBUTING), 'test_files', 'core_logic_folders' (folders containing primary source code), 'utility_folders', and 'data_related_files'. "
        f"Return ONLY a valid JSON object with these keys, where each value is a list of relevant file/folder paths. If a category is empty, return an empty list []."
    )
    response = await llm.aask(prompt)
    try:
        summary = json.loads(response)
        # Basic validation
        expected_keys = ['main_entrypoints', 'configuration_files', 'documentation_files', 'test_files', 'core_logic_folders', 'utility_folders', 'data_related_files']
        for key in expected_keys:
            if key not in summary or not isinstance(summary[key], list):
                 summary[key] = [] # Ensure keys exist and are lists
        return summary
    except Exception as e:
        logger.error(f"Error parsing structure summary JSON: {e}. Raw: {response}")
        return {"error": f"LLM did not return valid JSON for structure. {e}", "llm_raw_response": response}

async def compute_architecture(files: List[str], llm: LLMService) -> str:
    """ Uses LLM for a natural language architecture summary. """
    logger.info("Computing architecture summary...")
    if not files: return "No files found to analyze."
    files_sample = files[:MAX_FILES_FOR_PROMPT]
    prompt = (
        f"Based on this file list:\n{json.dumps(files_sample, indent=2)}\n\n"
        f"Provide a concise (2-3 sentences) high-level summary of the likely software architecture. What type of application is it? What are its main components?"
    )
    response = await llm.aask(prompt)
    return response

async def compute_complexity(files: List[str], llm: LLMService) -> str:
    """ Uses LLM to estimate project complexity. """
    logger.info("Computing complexity estimate...")
    if not files: return "Complexity: Unknown (No files found)."
    files_sample = files[:MAX_FILES_FOR_PROMPT]
    prompt = (
        f"Estimate the complexity of a codebase with these files:\n{json.dumps(files_sample, indent=2)}\n\n"
        f"Consider file count ({len(files)} total), apparent tech stack diversity, and potential component interactions. Provide a rating (Low, Medium, High, Very High) and a 1-sentence justification."
    )
    response = await llm.aask(prompt)
    return response

# --- Tool: Read File Content (Needed by other agents) ---
def read_file_content(project_path: str, relative_filepath: str) -> str:
    """ Reads text content of a specific file. Returns error string on failure. """
    # logger.debug(f"Reading file: {relative_filepath}") # Debug level
    full_path = os.path.normpath(os.path.join(project_path, relative_filepath)) # Normalize path

    # Security check: Ensure the path is still within the project directory
    if not full_path.startswith(os.path.normpath(project_path)):
        logger.error(f"Attempted path traversal: {relative_filepath}")
        return "Error: Access denied (Path Traversal)."

    try:
        if not os.path.exists(full_path): return f"Error: File not found at '{relative_filepath}'."
        if os.path.getsize(full_path) > MAX_FILE_SIZE_BYTES: return f"Error: File '{relative_filepath}' too large."
        if is_binary_file(full_path): return f"Error: Cannot read binary file '{relative_filepath}'."

        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file '{relative_filepath}': {e}")
        return f"Error reading file '{relative_filepath}': {e}"
```

-----

### 3\. `tools/file_proc_tools.py` (Completed)

Tools for analyzing individual file contents.

````python
# --- tools/file_proc_tools.py ---

from typing import List, Dict, Any
from services.llm_service import LLMService
import json
import logging

# --- Constants ---
MAX_CONTENT_FOR_PROMPT = 70000 # Max characters per file content in prompt (adjust based on LLM context window)

# --- Logging ---
logger = logging.getLogger(__name__)

# --- LLM-Powered Tools ---
async def extract_code_elements(filename: str, file_content: str, llm: LLMService) -> List[Dict[str, Any]]:
    """ Uses LLM to extract functions, classes, and roles from a single file's content. """
    logger.info(f"Extracting elements from: {filename}")
    if not file_content or file_content.startswith("Error:"):
        return [{"file": filename, "error": "Invalid or missing content provided."}]
    if len(file_content) > MAX_CONTENT_FOR_PROMPT:
        logger.warning(f"Content for {filename} truncated for element extraction.")
        file_content = file_content[:MAX_CONTENT_FOR_PROMPT] + "\n... [TRUNCATED]"

    prompt = (
        f"Analyze source code from '{filename}':\n```\n{file_content}\n```\n\n"
        f"Identify functions, classes, methods, or important config blocks. "
        f"Return ONLY a valid JSON list of objects. Each object must have keys: 'name'(string), 'type'(string, e.g., 'function', 'class', 'method', 'config'), 'start_line'(int, approximate), 'description'(string, 1 concise sentence)."
    )
    response = await llm.aask(prompt)
    try:
        elements = json.loads(response)
        if not isinstance(elements, list): raise ValueError("Expected a list")
        # Add filename and validate structure minimally
        validated_elements = []
        for i, el in enumerate(elements):
            if isinstance(el, dict) and 'name' in el and 'type' in el and 'start_line' in el and 'description' in el:
                el['file'] = filename
                validated_elements.append(el)
            else: logger.warning(f"Skipping invalid element structure in {filename}: {el}")
        return validated_elements
    except Exception as e:
        logger.error(f"Error parsing elements JSON for {filename}: {e}. Raw: {response}")
        return [{"file": filename, "error": f"LLM did not return valid JSON list for elements. {e}", "llm_raw_response": response}]

async def extract_dependencies(filename: str, file_content: str, llm: LLMService) -> Dict[str, Any]:
    """ Uses LLM to identify dependencies (imports, requires) in a single file's content. """
    logger.info(f"Extracting dependencies from: {filename}")
    if not file_content or file_content.startswith("Error:"):
        return {"file": filename, "error": "Invalid or missing content provided.", "dependencies": []}
    if len(file_content) > MAX_CONTENT_FOR_PROMPT:
        logger.warning(f"Content for {filename} truncated for dependency extraction.")
        file_content = file_content[:MAX_CONTENT_FOR_PROMPT] + "\n... [TRUNCATED]"

    prompt = (
        f"Analyze source code from '{filename}':\n```\n{file_content}\n```\n\n"
        f"List all imported libraries, modules, or external dependencies (e.g., 'requests', 'express', './utils'). "
        f"Return ONLY a valid JSON object with one key 'dependencies' containing a list of strings."
    )
    response = await llm.aask(prompt)
    try:
        deps_data = json.loads(response)
        if isinstance(deps_data, dict) and 'dependencies' in deps_data and isinstance(deps_data['dependencies'], list):
             deps_data['file'] = filename # Add filename for context
             return deps_data
        else: raise ValueError("Expected JSON object with 'dependencies' list.")
    except Exception as e:
        logger.error(f"Error parsing dependencies JSON for {filename}: {e}. Raw: {response}")
        # Fallback parsing
        deps_list = [d.strip() for d in response.split('\n') if d.strip() and len(d.strip()) > 1]
        return {"file": filename, "dependencies": deps_list, "error": f"LLM did not return valid JSON. {e}", "llm_raw_response": response}

# Note: extract_file_relationships is complex and costly. Consider simplifying or running selectively.
# It requires passing content of MULTIPLE files.
async def extract_file_relationships(filenames: List[str], file_contents: Dict[str, str], llm: LLMService) -> List[Dict[str, Any]]:
    """ Uses LLM to identify relationships (imports, calls) between multiple files. """
    logger.info(f"Extracting relationships between {len(filenames)} files...")
    if len(filenames) < 2: return [] # Need at least two files

    # Prepare content snippets for the prompt, respecting limits
    prompt_content = ""
    total_len = 0
    MAX_REL_CONTENT = 60000 # Adjust as needed
    MAX_SNIPPET_LEN = 3000

    processed_files = []
    for fname in filenames:
        content = file_contents.get(fname, "")
        if content.startswith("Error:") or not content: continue # Skip error or empty content

        snippet = content[:MAX_SNIPPET_LEN] + ("..." if len(content) > MAX_SNIPPET_LEN else "")
        segment = f"\n--- File: {fname} ---\n```\n{snippet}\n```\n"

        if total_len + len(segment) > MAX_REL_CONTENT:
            logger.warning(f"Stopping relationship analysis content inclusion at {len(processed_files)} files due to length limit.")
            break
        prompt_content += segment
        total_len += len(segment)
        processed_files.append(fname)

    if len(processed_files) < 2:
        return [{"error": "Not enough valid file content provided for relationship analysis."}]

    prompt = (
        f"Analyze relationships between these files based on content snippets. Identify imports or potential function/class usage between them.\n"
        f"{prompt_content}\n\n"
        f"Return ONLY a valid JSON list of objects. Each object must have keys: "
        f"'source_file'(string), 'target_file'(string), 'type'(string, e.g., 'import', 'call'), 'details'(string, optional)."
    )
    response = await llm.aask(prompt)
    try:
        relationships = json.loads(response)
        if not isinstance(relationships, list): raise ValueError("Expected a list")
        # Basic validation of items
        valid_rels = [r for r in relationships if isinstance(r,dict) and 'source_file' in r and 'target_file' in r and 'type' in r]
        if len(valid_rels) != len(relationships): logger.warning("Some relationship items had invalid structure.")
        return valid_rels
    except Exception as e:
        logger.error(f"Error parsing relationships JSON: {e}. Raw: {response}")
        return [{"error": f"LLM did not return valid JSON list for relationships. {e}", "llm_raw_response": response}]

async def suggest_skip_patterns(files: List[str], llm: LLMService) -> List[str]:
    """ Uses LLM to suggest additional skip patterns based on file list. """
    logger.info("Suggesting skip patterns...")
    if not files: return []
    files_sample = files[:MAX_FILES_FOR_PROMPT]
    prompt = (
        f"Based on file paths: {json.dumps(files_sample)}\nAre there other common patterns for directories/file types to exclude from analysis (build artifacts, logs, compiled code, assets)? List potential patterns (like '*.log', 'temp/'). Concise list."
    )
    response = await llm.aask(prompt)
    try: # Basic parsing
        if response.strip().startswith('['):
            import ast; patterns = ast.literal_eval(response.strip()); return patterns if isinstance(patterns, list) else [response]
        return [p.strip() for p in response.split('\n') if p.strip() and len(p.strip()) > 1]
    except: return [response]
````

-----

### 4\. `tools/code_discovery_tools.py` (Completed)

Tools for semantic chunking, enrichment, and persona insights.

````python
# --- tools/code_discovery_tools.py ---

from typing import List, Dict, Any
from services.llm_service import LLMService
import json
import logging

# --- Constants ---
MAX_CONTENT_FOR_PROMPT_CHUNK = 80000 # Larger limit for chunking itself
MAX_CHUNKS_FOR_PROMPT = 50
MAX_CONTENT_PREVIEW = 300 # Chars for preview in enrichment/persona prompts

# --- Logging ---
logger = logging.getLogger(__name__)

# --- LLM-Powered Tools ---
async def chunk_code_semantically(filename: str, file_content: str, llm: LLMService) -> List[Dict[str, Any]]:
    """ Uses LLM to chunk a single file's content into logical semantic units. """
    logger.info(f"Semantically chunking: {filename}")
    if not file_content or file_content.startswith("Error:"):
        return [{"file": filename, "error": "Invalid or missing content for chunking."}]
    content_to_chunk = file_content
    if len(file_content) > MAX_CONTENT_FOR_PROMPT_CHUNK:
        logger.warning(f"Content for {filename} truncated for semantic chunking.")
        content_to_chunk = file_content[:MAX_CONTENT_FOR_PROMPT_CHUNK] + "\n... [TRUNCATED]"

    prompt = (
        f"Analyze source code from '{filename}':\n```\n{content_to_chunk}\n```\n\n"
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
        chunks = json.loads(response)
        if not isinstance(chunks, list): raise ValueError("Expected a list")
        validated = []
        for i, ch in enumerate(chunks):
            if isinstance(ch,dict) and all(k in ch for k in ['type','name','start_line','end_line','content','summary']):
                ch['file'] = filename; ch['chunk_id'] = f"{filename}_{i}" # Add unique ID
                validated.append(ch)
            else: logger.warning(f"Skipping invalid chunk structure in {filename}: {ch}")
        return validated
    except Exception as e:
        logger.error(f"Error parsing semantic chunks JSON for {filename}: {e}. Raw: {response}")
        # Fallback: single chunk for the whole file
        return [{"file": filename, "chunk_id": f"{filename}_0", "type": "file", "name": filename, "start_line": 1, "end_line": file_content.count('\n')+1, "content": file_content, "summary": "Error during chunking.", "error": str(e), "llm_raw_response": response}]


async def enrich_chunks_with_metadata(chunks: List[Dict[str, Any]], llm: LLMService) -> List[Dict[str, Any]]:
    """ Uses LLM to add role, dependencies, and business context metadata to chunks. """
    logger.info(f"Enriching {len(chunks)} chunks with metadata...")
    if not chunks: return []

    # Process in batches if needed (example with one batch)
    chunks_batch = chunks[:MAX_CHUNKS_FOR_PROMPT]
    if len(chunks) > MAX_CHUNKS_FOR_PROMPT: logger.warning(f"Processing only first {MAX_CHUNKS_FOR_PROMPT} chunks for enrichment.")

    prompt_chunks = []
    for i, chunk in enumerate(chunks_batch):
        prompt_chunks.append({
            "id": chunk.get('chunk_id', f"chunk_{i}"), # Use existing ID if available
            "file": chunk.get('file', '?'), "type": chunk.get('type', '?'), "name": chunk.get('name', '?'),
            "summary": chunk.get('summary', ''),
            "content_preview": chunk.get('content', '')[:MAX_CONTENT_PREVIEW] + "..."
        })

    prompt = (
        f"Analyze code chunks:\n{json.dumps(prompt_chunks, indent=2)}\n\n"
        f"For each chunk (by 'id'), determine 'code_role' (e.g., 'API Endpoint', 'Business Logic', 'Data Model', 'Utility', 'Config', 'Test'), main 'dependencies' (libs/modules from preview/summary), and brief 'business_context' (1 sentence relevance/risk).\n"
        f"Return ONLY a valid JSON list. Each object must contain 'id' and the new keys: 'code_role'(string), 'dependencies'(list[string]), 'business_context'(string)."
    )
    response = await llm.aask(prompt)
    try:
        enrichment_data = json.loads(response)
        if not isinstance(enrichment_data, list): raise ValueError("Expected list")
        enrichment_map = {item.get('id'): item for item in enrichment_data if isinstance(item,dict) and 'id' in item}

        for chunk in chunks_batch:
            chunk_id = chunk.get('chunk_id')
            enrichment = enrichment_map.get(chunk_id) if chunk_id else None
            if enrichment:
                chunk['code_role'] = enrichment.get('code_role', 'Unknown')
                chunk['dependencies'] = enrichment.get('dependencies', [])
                chunk['business_context'] = enrichment.get('business_context', 'N/A')
            else: # Handle missing/failed enrichment
                chunk['code_role'] = 'Error'; chunk['dependencies'] = []; chunk['business_context'] = 'Enrichment failed.'
        # If processing all chunks, extend results here. For now, just return the processed batch.
        # If chunks > MAX_CHUNKS_FOR_PROMPT, remaining chunks won't be enriched.
        return chunks_batch + chunks[MAX_CHUNKS_FOR_PROMPT:] # Add unprocessed back if any

    except Exception as e:
        logger.error(f"Error parsing enrichment JSON: {e}. Raw: {response}")
        for chunk in chunks_batch: chunk['code_role'] = 'Error'; chunk['dependencies'] = []; chunk['business_context'] = f'Enrichment Error: {e}'
        return chunks # Return original chunks on error

async def generate_persona_insights(chunks: List[Dict[str, Any]], llm: LLMService, persona: str) -> List[Dict[str, Any]]:
    """ Uses LLM to generate persona-specific insights (SDE, PM) for enriched chunks. """
    logger.info(f"Generating insights for persona: {persona} on {len(chunks)} chunks...")
    if not chunks: return []

    chunks_batch = chunks[:MAX_CHUNKS_FOR_PROMPT]
    if len(chunks) > MAX_CHUNKS_FOR_PROMPT: logger.warning(f"Processing only first {MAX_CHUNKS_FOR_PROMPT} chunks for persona insights.")

    prompt_chunks = []
    for i, chunk in enumerate(chunks_batch):
        prompt_chunks.append({ # Send relevant enriched data
            "id": chunk.get('chunk_id', f"chunk_{i}"),
            "file": chunk.get('file','?'), "type": chunk.get('type','?'), "name": chunk.get('name','?'),
            "summary": chunk.get('summary','?'), "code_role": chunk.get('code_role','?'),
            "dependencies": chunk.get('dependencies',[]), "business_context": chunk.get('business_context','?'),
        })

    if persona.lower() == 'sde': persona_instr = "Focus on technical complexity, dependencies, testability, refactoring needs for a Software Dev Engineer."
    elif persona.lower() == 'pm': persona_instr = "Focus on business logic, user features, roadmap impact, risks for a Product Manager."
    else: persona_instr = "Provide a general overview."

    prompt = (
        f"Analyze enriched code chunks for persona '{persona}':\n{json.dumps(prompt_chunks, indent=2)}\n\n"
        f"For each chunk (by 'id'), provide a concise 'persona_insight' (1-2 sentences). {persona_instr}\n"
        f"Return ONLY a valid JSON list. Each object must contain 'id' and 'persona_insight'(string)."
    )
    response = await llm.aask(prompt)
    try:
        insight_data = json.loads(response)
        if not isinstance(insight_data, list): raise ValueError("Expected list")
        insight_map = {item.get('id'): item.get('persona_insight', 'N/A') for item in insight_data if isinstance(item,dict) and 'id' in item}

        insight_key = f'insight_{persona.lower()}'
        for chunk in chunks_batch:
            chunk_id = chunk.get('chunk_id')
            chunk[insight_key] = insight_map.get(chunk_id, 'Failed to generate insight.')

        # If processing all chunks, extend results here.
        return chunks_batch + chunks[MAX_CHUNKS_FOR_PROMPT:] # Add unprocessed back

    except Exception as e:
        logger.error(f"Error parsing persona insight JSON ({persona}): {e}. Raw: {response}")
        insight_key = f'insight_{persona.lower()}'
        for chunk in chunks_batch: chunk[insight_key] = f'Insight Error: {e}'
        return chunks # Return original chunks on error

````

-----

### 5\. `agents/repo_intel_agent.py` (Completed)

Calls the repo intelligence tools.

```python
# --- agents/repo_intel_agent.py ---

import os
from typing import Dict, Any, List
# Adjust import path
from tools.repo_intel_tools import (
    list_project_files,
    detect_tech_stack,
    summarize_structure,
    compute_architecture,
    compute_complexity
)
from services.llm_service import LLMService

class RepoIntelligenceAgent:
    """ Agent for initial codebase intelligence. """
    def __init__(self, llm_api_key: str | None = None):
        self.llm = LLMService(api_key=llm_api_key)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """ Runs the repo intelligence tools. """
        print("\n--- Running Repo Intelligence Agent ---")
        project_path = state.get("project_path")
        if not project_path or not os.path.isdir(project_path):
            state["repo_intel_error"] = "Project path missing or invalid."
            return state

        # 1. List files (Sync)
        repo_files = list_project_files(project_path)
        if not repo_files:
            state["repo_intel_error"] = "No relevant files found in the project path."
            state["repo_files"] = []
            return state # Stop if no files found
        state["repo_files"] = repo_files # Update state early

        # 2. Run async LLM tools
        try:
            tech_stack = await detect_tech_stack(repo_files, self.llm)
            structure = await summarize_structure(repo_files, self.llm)
            architecture = await compute_architecture(repo_files, self.llm)
            complexity = await compute_complexity(repo_files, self.llm)
        except Exception as e:
            # Catch potential errors during async calls
            print(f"Error during repo intel async tool execution: {e}")
            state["repo_intel_error"] = f"LLM analysis failed: {e}"
            # Still return partial results if available
            state["tech_stack"] = state.get("tech_stack", ["Error"])
            state["structure_summary"] = state.get("structure_summary", {"error": "LLM failed"})
            state["architecture_summary"] = state.get("architecture_summary", "Error")
            state["complexity_assessment"] = state.get("complexity_assessment", "Error")
            return state


        # 3. Update state with results
        state["tech_stack"] = tech_stack
        state["structure_summary"] = structure
        state["architecture_summary"] = architecture
        state["complexity_assessment"] = complexity
        state["repo_intel_error"] = None # Clear error if successful

        print("--- Finished Repo Intelligence Agent ---")
        return state
```

-----

### 6\. `agents/file_proc_agent.py` (Completed)

Reads files and calls file processing tools.

```python
# --- agents/file_proc_agent.py ---

import os
from typing import Dict, Any, List
# Adjust import paths
from tools.file_proc_tools import (
    extract_code_elements,
    extract_dependencies,
    extract_file_relationships,
    suggest_skip_patterns,
)
from tools.repo_intel_tools import read_file_content # Use the reader tool
from services.llm_service import LLMService
import asyncio
import logging

logger = logging.getLogger(__name__)
# Limit how many files to process in detail
MAX_FILES_TO_PROCESS = 50

class FileProcessingAgent:
    """ Agent for file-level analysis: elements, dependencies, relationships. """
    def __init__(self, llm_api_key: str | None = None):
        self.llm = LLMService(api_key=llm_api_key)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """ Runs file processing tools on relevant files. """
        print("\n--- Running File Processing Agent ---")
        project_path = state.get("project_path")
        repo_files = state.get("repo_files", []) # Get files from previous agent

        if not project_path or not repo_files:
            state["file_proc_error"] = "Project path or file list missing."
            return state
        if state.get("repo_intel_error"): # Skip if previous agent failed
            state["file_proc_error"] = "Skipped due to repo intel error."
            return state

        files_to_process = repo_files[:MAX_FILES_TO_PROCESS]
        if len(repo_files) > MAX_FILES_TO_PROCESS: logger.warning(f"Processing details for first {MAX_FILES_TO_PROCESS} files.")

        # --- Read file contents ---
        file_contents: Dict[str, str] = {}
        print(f"Reading content for {len(files_to_process)} files...")
        read_errors = 0
        for rel_path in files_to_process:
            content = read_file_content(project_path, rel_path) # Sync read
            if content.startswith("Error:"):
                 logger.warning(f"Read Error for {rel_path}: {content}")
                 read_errors += 1
            file_contents[rel_path] = content
        print(f"Finished reading files ({read_errors} errors).")

        # --- Run async tools ---
        element_tasks = []
        dependency_tasks = []
        valid_content_files = [] # Files with valid content for relationship analysis
        for rel_path, content in file_contents.items():
            if not content.startswith("Error:"):
                element_tasks.append(extract_code_elements(rel_path, content, self.llm))
                dependency_tasks.append(extract_dependencies(rel_path, content, self.llm))
                valid_content_files.append(rel_path)

        # Only run relationship if there's enough valid content
        if len(valid_content_files) >= 2:
            relationship_task = extract_file_relationships(valid_content_files, file_contents, self.llm)
        else:
            relationship_task = asyncio.sleep(0, result=[]) # Return empty list immediately
            logger.warning("Skipping relationship analysis: Not enough valid files.")

        skip_pattern_task = suggest_skip_patterns(repo_files, self.llm)

        # Gather results
        try:
            print("Gathering file processing results...")
            code_elements_results = await asyncio.gather(*element_tasks)
            dependencies_results = await asyncio.gather(*dependency_tasks)
            file_relationships = await relationship_task
            skip_patterns = await skip_pattern_task
            print("Finished gathering results.")
        except Exception as e:
            logger.error(f"Error during file processing async tool execution: {e}")
            state["file_proc_error"] = f"Async tool execution failed: {e}"
            # Populate state with error indicators
            state["code_elements"] = [{"error": str(e)}]
            state["dependencies"] = [{"error": str(e)}]
            state["file_relationships"] = [{"error": str(e)}]
            state["suggested_skip_patterns"] = [f"Error: {e}"]
            return state


        # Flatten list of lists for elements and filter errors reported by tool
        all_code_elements = [el for sublist in code_elements_results for el in sublist if not el.get("error")]

        # Filter errors reported by dependency tool
        valid_dependencies = [dep for dep in dependencies_results if not dep.get("error")]

        # Filter errors reported by relationship tool
        valid_relationships = [rel for rel in file_relationships if not rel.get("error")]

        # Update state
        state["file_contents"] = file_contents # Store contents if needed downstream
        state["code_elements"] = all_code_elements
        state["dependencies"] = valid_dependencies
        state["file_relationships"] = valid_relationships
        state["suggested_skip_patterns"] = skip_patterns
        state["file_proc_error"] = None # Clear error if successful

        print("--- Finished File Processing Agent ---")
        return state
```

-----

### 7\. `agents/code_discovery_agent.py` (Completed)

Runs semantic chunking and persona insights.

```python
# --- agents/code_discovery_agent.py ---

import os
from typing import Dict, Any, List
# Adjust import paths
from tools.code_discovery_tools import (
    chunk_code_semantically,
    enrich_chunks_with_metadata,
    generate_persona_insights,
)
# Use file contents read by previous agent
# from tools.repo_intel_tools import read_file_content
from services.llm_service import LLMService
import asyncio
import logging

logger = logging.getLogger(__name__)
# Limit files for expensive semantic chunking
MAX_FILES_FOR_SEMANTIC_CHUNK = 25 # Reduced limit

class CodeDiscoveryAgent:
    """ Agent for semantic chunking, enrichment, and persona insights. """
    def __init__(self, llm_api_key: str | None = None):
        self.llm = LLMService(api_key=llm_api_key)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """ Runs semantic chunking and analysis tools. """
        print("\n--- Running Code Discovery Agent ---")
        project_path = state.get("project_path") # Still needed? Maybe not if content passed
        # Use files/content from previous state
        repo_files = state.get("repo_files", [])
        file_contents = state.get("file_contents", {})
        personas = state.get("personas", ["SDE", "PM"]) # Get target personas

        if not file_contents: # Check if content is available
            state["code_discovery_error"] = "File contents missing from state."
            # Check if previous agent had error
            if state.get("file_proc_error"):
                 state["code_discovery_error"] += f" Likely due to previous error: {state['file_proc_error']}"
            return state
        if state.get("file_proc_error"): # Skip if previous agent failed
             state["code_discovery_error"] = "Skipped due to file processing error."
             return state

        # --- Semantic Chunking ---
        # Select files to chunk (use only those with valid content)
        files_to_chunk = [f for f, c in file_contents.items() if not c.startswith("Error:")]
        files_to_chunk = files_to_chunk[:MAX_FILES_FOR_SEMANTIC_CHUNK] # Apply limit
        if len(file_contents) > MAX_FILES_FOR_SEMANTIC_CHUNK:
             logger.warning(f"Semantically chunking first {MAX_FILES_FOR_SEMANTIC_CHUNK} valid files.")

        print(f"Starting semantic chunking for {len(files_to_chunk)} files...")
        chunking_tasks = []
        for rel_path in files_to_chunk:
            content = file_contents[rel_path] # We know it's valid here
            chunking_tasks.append(chunk_code_semantically(rel_path, content, self.llm))

        try:
            chunk_results = await asyncio.gather(*chunking_tasks)
            all_chunks = [chunk for sublist in chunk_results for chunk in sublist if not chunk.get("error")]
            print(f"Generated {len(all_chunks)} semantic chunks.")
        except Exception as e:
            logger.error(f"Error during semantic chunking gathering: {e}")
            state["code_discovery_error"] = f"Semantic chunking failed: {e}"
            state["semantic_chunks"] = [{"error": str(e)}]
            return state


        if not all_chunks:
             state["semantic_chunks"] = []
             print("No semantic chunks generated.")
             print("--- Finished Code Discovery Agent ---")
             return state

        # --- Enrich Chunks ---
        try:
            print("Enriching chunks with metadata...")
            enriched_chunks = await enrich_chunks_with_metadata(all_chunks, self.llm)
        except Exception as e:
             logger.error(f"Error during chunk enrichment: {e}")
             state["code_discovery_error"] = f"Chunk enrichment failed: {e}"
             state["semantic_chunks"] = all_chunks # Store un-enriched chunks
             return state


        # --- Generate Persona Insights ---
        try:
            print(f"Generating insights for personas: {personas}...")
            chunks_for_persona = enriched_chunks # Start with enriched
            # Apply insights - modifying the list in place
            for persona in personas:
                # This function modifies chunks_for_persona list by adding insight keys
                await generate_persona_insights(chunks_for_persona, self.llm, persona)
            # The list now contains insights for all personas
            final_chunks_with_insights = chunks_for_persona
        except Exception as e:
            logger.error(f"Error during persona insight generation: {e}")
            state["code_discovery_error"] = f"Persona insight generation failed: {e}"
            # Store enriched but potentially insight-less chunks
            final_chunks_with_insights = enriched_chunks # Fallback
            # Add error marker to chunks maybe?
            for chunk in final_chunks_with_insights:
                 for p in personas: chunk[f"insight_{p.lower()}"] = f"Error: {e}"


        # Update state with final results
        state["semantic_chunks"] = final_chunks_with_insights
        state["code_discovery_error"] = None # Clear error if successful

        print("--- Finished Code Discovery Agent ---")
        return state
```

-----

### 8\. `graph/nodes.py` (Completed - Async)

Connects graph nodes to the agent `process` methods. Node functions are `async`.

```python
# --- graph/nodes.py ---

import os
from typing import Dict, Any
# Import agents (adjust paths if needed)
from agents.repo_intel_agent import RepoIntelligenceAgent
from agents.file_proc_agent import FileProcessingAgent
from agents.code_discovery_agent import CodeDiscoveryAgent
import logging

logger = logging.getLogger(__name__)
# Load API key once
LLM_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Node functions MUST be async ---

async def repo_intelligence_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """ Node that runs the RepoIntelligenceAgent. """
    logger.info("Executing Repo Intelligence Node")
    try:
        agent = RepoIntelligenceAgent(llm_api_key=LLM_API_KEY)
        # Use await to call the async process method
        result_state = await agent.process(state.copy()) # Pass a copy to avoid mutation issues?
        logger.info("Finished Repo Intelligence Node")
        # LangGraph expects nodes to return only the modified parts of the state.
        # However, since our agents update the state dict directly, we can return it.
        # If StateGraph uses operator.add, returning the full state is fine.
        # If it uses a simple update, returning only modified keys is safer.
        # Let's return the full state for simplicity, assuming StateGraph handles merging.
        return result_state
    except Exception as e:
        logger.error(f"Error in repo_intelligence_node: {e}", exc_info=True)
        # Return state with error indication
        state["repo_intel_error"] = f"Node failed: {e}"
        return state # Allow graph to potentially continue or handle error

async def file_processing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """ Node that runs the FileProcessingAgent. """
    logger.info("Executing File Processing Node")
    # Check if previous node failed
    if state.get("repo_intel_error"):
        logger.warning("Skipping File Processing due to previous error.")
        state["file_proc_error"] = "Skipped due to repo_intel_error"
        return state
    try:
        agent = FileProcessingAgent(llm_api_key=LLM_API_KEY)
        result_state = await agent.process(state.copy())
        logger.info("Finished File Processing Node")
        return result_state
    except Exception as e:
        logger.error(f"Error in file_processing_node: {e}", exc_info=True)
        state["file_proc_error"] = f"Node failed: {e}"
        return state

async def code_discovery_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """ Node that runs the CodeDiscoveryAgent. """
    logger.info("Executing Code Discovery Node")
    # Check if previous node failed
    if state.get("file_proc_error") or state.get("repo_intel_error"):
        logger.warning("Skipping Code Discovery due to previous error.")
        state["code_discovery_error"] = "Skipped due to previous error"
        return state
    try:
        agent = CodeDiscoveryAgent(llm_api_key=LLM_API_KEY)
        result_state = await agent.process(state.copy())
        logger.info("Finished Code Discovery Node")
        return result_state
    except Exception as e:
        logger.error(f"Error in code_discovery_node: {e}", exc_info=True)
        state["code_discovery_error"] = f"Node failed: {e}"
        return state
```

-----

### 9\. `graph/graph.py` (Completed)

Defines the state and wires the async nodes sequentially.

```python
# --- graph/graph.py ---

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import operator
import json
import asyncio
import os
import logging
from dotenv import load_dotenv

# Import the async node functions
from graph.nodes import (
    repo_intelligence_node,
    file_processing_node,
    code_discovery_node
)

logger = logging.getLogger(__name__)

# --- Define the State ---
# Includes all expected inputs and outputs from the agents
class CodeAnalysisState(TypedDict, total=False): # Use total=False for flexibility
    project_path: str                 # Input
    personas: List[str]               # Input
    # Repo Intel Outputs
    repo_files: List[str]
    tech_stack: List[str]
    structure_summary: Dict[str, Any]
    architecture_summary: str
    complexity_assessment: str
    # File Proc Outputs
    file_contents: Optional[Dict[str, str]] # Make optional, might be large
    code_elements: List[Dict[str, Any]]
    dependencies: List[Dict[str, Any]]
    file_relationships: List[Dict[str, Any]]
    suggested_skip_patterns: List[str]
    # Code Discovery Outputs
    semantic_chunks: List[Dict[str, Any]]
    # Error fields
    repo_intel_error: Optional[str]
    file_proc_error: Optional[str]
    code_discovery_error: Optional[str]

# --- Graph Building Function ---
# Make this a simple function, compile once when needed
analysis_graph_compiled = None

def build_analysis_graph() -> StateGraph:
    """ Builds and compiles the LangGraph analysis workflow. """
    global analysis_graph_compiled
    if analysis_graph_compiled:
        return analysis_graph_compiled # Return cached compiled graph

    print("[Graph] Building analysis graph...")
    graph_builder = StateGraph(CodeAnalysisState)

    # Add nodes using the async functions
    graph_builder.add_node("repo_intelligence", repo_intelligence_node)
    graph_builder.add_node("file_processing", file_processing_node)
    graph_builder.add_node("code_discovery", code_discovery_node)

    # Define the sequential flow
    graph_builder.set_entry_point("repo_intelligence")
    graph_builder.add_edge("repo_intelligence", "file_processing")
    graph_builder.add_edge("file_processing", "code_discovery")
    graph_builder.add_edge("code_discovery", END) # End after code discovery

    # Compile the graph
    analysis_graph_compiled = graph_builder.compile()
    print("[Graph] Analysis graph compiled.")
    try: print("\nGraph Structure:"); analysis_graph_compiled.get_graph().print_ascii()
    except Exception as e: print(f"Could not print graph: {e}")

    return analysis_graph_compiled

# --- Example Invocation (for testing graph.py directly) ---
if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    TEST_REPO_PATH = "../test_repo" # IMPORTANT: Adjust path relative to graph.py location
    if not os.path.isdir(TEST_REPO_PATH):
         print(f"ERROR: Test repository path '{os.path.abspath(TEST_REPO_PATH)}' not found.")
         exit()

    graph = build_analysis_graph() # Compile the graph

    initial_state: CodeAnalysisState = { "project_path": TEST_REPO_PATH, "personas": ["SDE", "PM"] }

    async def run_graph():
        print(f"\n--- Running analysis on: {TEST_REPO_PATH} ---")
        config = {"recursion_limit": 10}
        # Use ainvoke for final result after streaming
        final_state = await graph.ainvoke(initial_state, config=config)
        print("\n--- Graph Execution Complete ---")
        print("\nFinal State:")
        # Pretty print, handling potential non-serializable data gracefully
        print(json.dumps(final_state, indent=2, default=lambda o: f"<non-serializable: {type(o).__name__}>"))

    try: asyncio.run(run_graph())
    except Exception as e: print(f"\n--- Error: {e} ---"); import traceback; traceback.print_exc()

```

-----

### 10\. `api/routes.py` (Integration)

Add the new endpoint to trigger the analysis. We'll use FastAPI's `BackgroundTasks` for simplicity to run the graph without blocking the HTTP response.

```python
# --- api/routes.py (Additions) ---

import hashlib
import io
import os
import time # For unique folder names
from datetime import datetime, timedelta
import re
import zipfile
import json # For storing results
import logging # Use logging

# FastAPI imports
from fastapi import (
    APIRouter, Depends, File, HTTPException, UploadFile, status, BackgroundTasks # Add BackgroundTasks
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Other imports
import git
from sqlalchemy.orm import Session
from jose import JWTError, jwt

# Your project imports (adjust paths as needed)
# Assuming 'multi_agent_docs' is at the same level as 'api' or in PYTHONPATH
from multi_agent_docs.graph.graph import build_analysis_graph, CodeAnalysisState
# Database/Auth related imports (already present in your file)
from models.model import Initialisation, User, Project # Assuming Project model exists
from crud.crud import get_user_by_email, create_user, create_project, get_project_by_id, update_initialization_status, store_analysis_result # Add new CRUD functions
from database.db import get_db
from schema import schemas

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()
UPLOAD_BASE_DIR = "uploads"

# --- Existing code (JWT, security, User endpoints, Uploads) remains the same ---
# ... (copy all your existing code from the prompt here, including create_initialization etc.) ...

# JWT setup
SECRET_KEY = os.getenv("SECRET_KEY", "SECRET") # Use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

MAX_ZIP_SIZE = 100 * 1024 * 1024  # 100MB

GIT_URL_REGEX = re.compile(
    r"^(https://|git@|ssh://)[\w\.\-/:]+\.git$"
)

ALLOWED_EXTENSIONS = {'.py', '.js', '.java', '.c', '.cpp', '.ts', '.go', '.rb', '.php', '.cs', '.rs'}

# --- Helper Functions (create_initialization, has_code_files, is_repo_empty) ---
# ... (Keep these as they are) ...

security = HTTPBearer()

# --- Auth Functions (create_access_token, get_current_user) ---
# ... (Keep these as they are) ...

# --- User Endpoints (/users/, /token, /me) ---
# ... (Keep these as they are) ...

# --- Upload Endpoints (/upload/, /upload-git/) ---
# MODIFY upload functions to use a unique folder name and trigger background task
@router.post("/upload/", response_model=dict)
async def upload_zip(project_name: str, # Make endpoint async
                 persona: str,
                 zip_file: UploadFile = File(...),
                 db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user),
                 background_tasks: BackgroundTasks = Depends()): # Inject BackgroundTasks

    if persona not in ("SDE", "PM"):
        raise HTTPException(status_code=400, detail="Invalid persona.")
    if not zip_file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Must be a .zip file.")

    contents = await zip_file.read() # Use await for async read
    if len(contents) > MAX_ZIP_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 100MB limit.")

    # Create a unique directory using timestamp or UUID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_project_folder_name = f"{project_name}_{timestamp}"
    project_folder_path = os.path.abspath(os.path.join(UPLOAD_BASE_DIR, f"user_{current_user.id}", unique_project_folder_name))

    try:
        os.makedirs(project_folder_path, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(contents), 'r') as zip_ref:
            zip_ref.extractall(project_folder_path)
        logger.info(f"Extracted zip to: {project_folder_path}")
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP archive.")
    except Exception as e:
         logger.error(f"Error extracting zip: {e}")
         raise HTTPException(status_code=500, detail=f"Failed to extract zip file: {e}")

    # --- Database entries ---
    project = create_project(db, user_id=current_user.id, project_name=project_name, file_name=zip_file.filename, local_path=project_folder_path) # Store local path
    init = create_initialization(db, user_id=current_user.id, project_id=project.id, persona=persona) # Status is "started"

    # --- Trigger background analysis ---
    logger.info(f"Adding analysis task to background for init_id: {init.id}")
    background_tasks.add_task(
        run_analysis_background,
        db_session_factory=get_db.dependency, # Pass function to get session in background
        initialization_id=init.id,
        project_path=project_folder_path,
        personas=[persona] # Pass persona as list
        )

    return {
        "msg": f"Upload complete. Analysis started in background for persona '{persona}'. Check status later.",
        "project_id": project.id,
        "initialization_id": init.id,
        "status": init.status, # Initial status: "started"
        "persona": persona
    }


@router.post("/upload-git/", response_model=dict)
async def upload_git_repo( # Make endpoint async
    project_name: str,
    persona: str,
    git_url: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = Depends() # Inject BackgroundTasks
):
    if persona not in ("SDE", "PM"):
        raise HTTPException(status_code=400, detail="Invalid persona.")
    if not GIT_URL_REGEX.match(git_url):
        raise HTTPException(status_code=400, detail="Invalid git URL.")

    # --- Create unique directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_project_folder_name = f"{project_name}_{timestamp}"
    project_folder_path = os.path.abspath(os.path.join(UPLOAD_BASE_DIR, f"user_{current_user.id}", unique_project_folder_name))

    try:
        logger.info(f"Cloning {git_url} to {project_folder_path}")
        git.Repo.clone_from(git_url, project_folder_path) # This is blocking, consider async library or run_in_executor
        logger.info("Cloning complete.")
    except Exception as e:
        logger.error(f"Error cloning repo: {e}")
        raise HTTPException(status_code=400, detail=f"Error cloning repo: {e}")

    if is_repo_empty(project_folder_path):
        raise HTTPException(status_code=400, detail="Repo is empty.")
    if not has_code_files(project_folder_path):
        raise HTTPException(status_code=400, detail="Repo has no recognizable code files.")

    # --- Database entries ---
    project = create_project(db, user_id=current_user.id, project_name=project_name, file_name=git_url, local_path=project_folder_path) # Store path
    init = create_initialization(db, user_id=current_user.id, project_id=project.id, persona=persona)

    # --- Trigger background analysis ---
    logger.info(f"Adding analysis task to background for init_id: {init.id}")
    background_tasks.add_task(
        run_analysis_background,
        db_session_factory=get_db.dependency, # Pass factory
        initialization_id=init.id,
        project_path=project_folder_path,
        personas=[persona]
        )

    return {
        "msg": "Clone complete. Analysis started in background. Check status later.",
        "project_id": project.id,
        "initialization_id": init.id,
        "status": init.status,
        "persona": persona
    }


# --- NEW: Function to run analysis in background ---
# This function will be executed by BackgroundTasks
async def run_analysis_background(db_session_factory, initialization_id: int, project_path: str, personas: List[str]):
    """ Compiles and runs the LangGraph analysis, updates DB status/results. """
    db: Session = next(db_session_factory()) # Get a new session for the background task
    try:
        logger.info(f"[Background Task {initialization_id}] Starting analysis for path: {project_path}")
        update_initialization_status(db, initialization_id, "processing")

        # Compile graph (ensure build_analysis_graph is accessible)
        # It's better to compile once globally if possible, but for simplicity:
        try:
             analysis_graph = build_analysis_graph() # This compiles the graph
             if not analysis_graph:
                 raise RuntimeError("Failed to compile analysis graph.")
        except Exception as compile_err:
             logger.error(f"[Background Task {initialization_id}] Graph compilation failed: {compile_err}", exc_info=True)
             update_initialization_status(db, initialization_id, "failed", f"Graph compilation error: {compile_err}")
             return # Stop processing

        # Prepare initial state for the graph run
        initial_state = CodeAnalysisState(
            project_path=project_path,
            personas=personas
            # Graph nodes will populate the rest
        )
        config = {"recursion_limit": 15} # Adjust limit as needed

        # Run the graph asynchronously
        logger.info(f"[Background Task {initialization_id}] Invoking graph...")
        final_state = await analysis_graph.ainvoke(initial_state, config=config)
        logger.info(f"[Background Task {initialization_id}] Graph execution finished.")

        # Check for errors reported in the state
        final_status = "completed"
        error_msg = None
        if final_state.get("code_discovery_error"): error_msg = final_state["code_discovery_error"]
        elif final_state.get("file_proc_error"): error_msg = final_state["file_proc_error"]
        elif final_state.get("repo_intel_error"): error_msg = final_state["repo_intel_error"]

        if error_msg:
             final_status = "failed"
             logger.error(f"[Background Task {initialization_id}] Analysis failed: {error_msg}")
             update_initialization_status(db, initialization_id, final_status, error_msg)
        else:
             # Store results (convert state dict to JSON string)
             try:
                 result_json = json.dumps(final_state, default=str) # Handle non-serializable data
                 store_analysis_result(db, initialization_id, result_json)
                 update_initialization_status(db, initialization_id, final_status)
                 logger.info(f"[Background Task {initialization_id}] Analysis completed and results stored.")
             except Exception as store_err:
                  logger.error(f"[Background Task {initialization_id}] Failed to store results: {store_err}", exc_info=True)
                  update_initialization_status(db, initialization_id, "failed", f"Result storage error: {store_err}")

    except Exception as e:
        logger.error(f"[Background Task {initialization_id}] Unexpected error during analysis: {e}", exc_info=True)
        update_initialization_status(db, initialization_id, "failed", f"Unexpected background error: {e}")
    finally:
        db.close() # Close the session used by the background task


# --- NEW: Endpoint to check analysis status/results ---
@router.get("/analysis-status/{initialization_id}", response_model=schemas.InitializationOut) # Use your schema
def get_analysis_status(initialization_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """ Gets the status and potentially results of an analysis initialization. """
    init = db.query(Initialisation).filter(Initialisation.id == initialization_id).first()

    if not init:
        raise HTTPException(status_code=404, detail="Initialization record not found.")
    # Optional: Check if the init record belongs to the current user
    if init.user_id != current_user.id:
         raise HTTPException(status_code=403, detail="Not authorized to view this analysis.")

    # Try to parse results if completed
    results_dict = None
    if init.status == "completed" and init.results:
        try:
            results_dict = json.loads(init.results)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse results JSON for init_id {initialization_id}")
            results_dict = {"error": "Failed to parse results JSON"} # Or return raw string

    # Return using Pydantic schema (adjust InitializationOut schema if needed)
    return schemas.InitializationOut(
        id=init.id,
        project_id=init.project_id,
        user_id=init.user_id,
        persona=init.persona,
        status=init.status,
        error_message=init.error_message,
        created_at=init.created_at,
        updated_at=init.updated_at,
        results=results_dict # Pass parsed results or None
    )

```

**Explanation of Changes in `api/routes.py`:**

1.  **Imports:** Added `BackgroundTasks` from FastAPI, `json`, `logging`, and imports for your graph (`build_analysis_graph`, `CodeAnalysisState`), and assumed new CRUD functions (`get_project_by_id`, `update_initialization_status`, `store_analysis_result`).
2.  **`upload_zip` / `upload_git_repo`:**
      * Made `async def` to allow `await zip_file.read()`.
      * Inject `BackgroundTasks`.
      * Generate a `unique_project_folder_name` (e.g., using timestamp) to avoid overwrites.
      * Store the absolute `project_folder_path` in the `Project` database record (you'll need to add a `local_path` column to your `Project` model and `create_project` function).
      * Call `background_tasks.add_task()` to schedule `run_analysis_background`. We pass the DB *session factory*, `initialization_id`, the full `project_folder_path`, and the selected `personas`.
      * Return immediately with `initialization_id` and "started" status.
3.  **`run_analysis_background` (New Async Function):**
      * This function runs *after* the HTTP request is finished.
      * It gets a **new database session** using the passed factory.
      * It updates the `Initialisation` status to "processing".
      * It **compiles the graph** by calling `build_analysis_graph()`.
      * It prepares the `initial_state` using `project_path` and `personas`.
      * It **runs the graph** using `await analysis_graph.ainvoke()`.
      * It checks the `final_state` for any error flags set by the graph nodes.
      * If no errors, it converts the `final_state` dictionary to a JSON string.
      * It calls `store_analysis_result` (new CRUD function needed) to save the JSON string to the `Initialisation` record (you'll need to add a `results` column, likely `TEXT` or `JSONB` type, to your `Initialisation` model).
      * It calls `update_initialization_status` (new CRUD function needed) to set the final status ("completed" or "failed") and store any error message.
      * It closes the database session.
4.  **`get_analysis_status` (New Endpoint):**
      * A simple `GET` endpoint to fetch an `Initialisation` record by its ID.
      * Checks if the record exists and belongs to the current user.
      * If the status is "completed" and results exist, it tries to parse the results JSON before returning.
      * Uses a Pydantic schema (`InitializationOut`, which you need to define) to structure the response.

-----

### 11\. Database & CRUD Updates (Needs Implementation)

You need to modify your database models and CRUD functions:

1.  **`models/model.py`:**
      * Add `local_path: Mapped[str]` to your `Project` model.
      * Add `results: Mapped[Optional[str]] = mapped_column(Text)` (or `JSONB` if using Postgres) to your `Initialisation` model.
      * Add `error_message: Mapped[Optional[str]] = mapped_column(String(500))` to `Initialisation`.
      * Ensure `status` in `Initialisation` can hold "started", "processing", "completed", "failed".
2.  **`crud/crud.py`:**
      * Modify `create_project` to accept and store `local_path`.
      * Add `get_project_by_id(db: Session, project_id: int) -> Optional[Project]`.
      * Add `update_initialization_status(db: Session, init_id: int, status: str, error_msg: Optional[str] = None)`: Fetches the `Initialisation` by ID and updates its `status`, `error_message`, and `updated_at`.
      * Add `store_analysis_result(db: Session, init_id: int, result_json: str)`: Fetches the `Initialisation` and updates its `results` field.
3.  **`schema/schemas.py`:**
      * Define `InitializationOut` Pydantic model to include `id`, `project_id`, `user_id`, `persona`, `status`, `error_message`, `created_at`, `updated_at`, and `results: Optional[Dict[str, Any]]` (or `Optional[str]` if you don't parse it in the endpoint).

-----

**How to Run:**

1.  Apply the database schema changes (e.g., using Alembic migrations).
2.  Implement the new CRUD functions.
3.  Define the `InitializationOut` schema.
4.  Ensure all agent/tool/graph files (`multi_agent_docs/`) are accessible from your FastAPI project (e.g., install it as a package, add to `PYTHONPATH`, or place it within the API project structure).
5.  Run your FastAPI application.
6.  Use the `/upload/` or `/upload-git/` endpoints. They will return quickly.
7.  Poll the `/analysis-status/{initialization_id}` endpoint until the status changes from "processing" to "completed" or "failed". The results will be included in the response when completed.
