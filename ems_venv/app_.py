# --- tools/file_proc_tools.py ---

from typing import List, Dict, Any
from services.llm_service import LLMService
import json

# Define MAX_CONTENT_FOR_PROMPT if passing multiple file contents
MAX_CONTENT_FOR_PROMPT = 50000 # Limit total characters sent

async def extract_code_elements(filename: str, file_content: str, llm: LLMService) -> List[Dict[str, Any]]:
    """ Uses LLM to extract functions, classes, and roles from a single file's content. """
    print(f"[File Proc Tool] Extracting elements from: {filename}")
    if len(file_content) > MAX_CONTENT_FOR_PROMPT:
        print(f"Warning: Content for {filename} truncated for element extraction.")
        file_content = file_content[:MAX_CONTENT_FOR_PROMPT] + "\n... [TRUNCATED]"

    prompt = (
        f"Analyze the following source code from the file '{filename}':\n```\n{file_content}\n```\n\n"
        f"Identify all major functions, classes, methods, or important configuration blocks. "
        f"For each element, provide its name, type (function, class, method, config), start line number (approximate if necessary), and a concise one-sentence description of its purpose. "
        f"Return the result as a JSON list of objects, each with keys: 'name', 'type', 'start_line', 'description'."
    )
    response = await llm.aask(prompt)
    try:
        elements = json.loads(response)
        # Add filename back to each element for context
        for el in elements:
            el['file'] = filename
        return elements
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON for elements in {filename}, returning raw.")
        return [{"file": filename, "llm_raw_response": response}]
    except Exception as e:
         print(f"Error parsing elements for {filename}: {e}")
         return [{"file": filename, "error": str(e), "llm_raw_response": response}]


async def extract_dependencies(filename: str, file_content: str, llm: LLMService) -> Dict[str, Any]:
    """ Uses LLM to identify dependencies (imports, requires) in a single file's content. """
    print(f"[File Proc Tool] Extracting dependencies from: {filename}")
    if len(file_content) > MAX_CONTENT_FOR_PROMPT:
        print(f"Warning: Content for {filename} truncated for dependency extraction.")
        file_content = file_content[:MAX_CONTENT_FOR_PROMPT] + "\n... [TRUNCATED]"

    prompt = (
        f"Analyze the following source code from '{filename}':\n```\n{file_content}\n```\n\n"
        f"List all imported libraries, modules, or external dependencies mentioned (e.g., 'import requests', 'require(\"express\")'). "
        f"If possible, include version numbers if specified nearby. "
        f"Return a JSON object with the key 'dependencies' containing a list of strings."
    )
    response = await llm.aask(prompt)
    try:
        # Expecting format like {"dependencies": ["requests", "os", "fastapi==0.1.0"]}
        deps = json.loads(response)
        # Add filename for context
        deps['file'] = filename
        return deps
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON for dependencies in {filename}, returning raw.")
        # Try a simple split as fallback
        deps_list = [d.strip() for d in response.split('\n') if d.strip() and not d.strip().startswith('{')]
        return {"file": filename, "dependencies": deps_list, "llm_raw_response": response}
    except Exception as e:
         print(f"Error parsing dependencies for {filename}: {e}")
         return {"file": filename, "error": str(e), "llm_raw_response": response}


async def extract_file_relationships(filenames: List[str], file_contents: Dict[str, str], llm: LLMService) -> List[Dict[str, Any]]:
    """
    Uses LLM to identify relationships (imports, calls) between multiple files based on their content.
    Expects a dictionary mapping filenames to their content.
    """
    print(f"[File Proc Tool] Extracting relationships between {len(filenames)} files...")

    # Prepare content for prompt, limiting total size
    prompt_content = ""
    total_len = 0
    included_files = []
    for fname in filenames:
        content = file_contents.get(fname, "")
        if not content: continue
        segment = f"\n--- File: {fname} ---\n```\n{content[:5000]}...\n```\n" # Limit content per file
        if total_len + len(segment) > MAX_CONTENT_FOR_PROMPT * 2: # Allow slightly more for relationships
             print(f"Warning: Stopping relationship analysis early due to total content length limit.")
             break
        prompt_content += segment
        total_len += len(segment)
        included_files.append(fname)

    if not included_files:
        return [{"error": "No content provided for relationship analysis."}]

    prompt = (
        f"Analyze the relationships between the following code files based on their content. "
        f"Identify which files import or call functions/classes from other files within this list. "
        f"Focus on direct dependencies revealed by import statements or function/method calls.\n"
        f"{prompt_content}\n\n"
        f"Return the relationships as a JSON list of objects, each with keys: "
        f"'source_file' (the file doing the importing/calling), "
        f"'target_file' (the file being imported/called), "
        f"'type' (e.g., 'import', 'function_call', 'class_instantiation'), "
        f"and 'details' (e.g., the specific function/class name involved)."
    )
    response = await llm.aask(prompt)
    try:
        relationships = json.loads(response)
        return relationships
    except json.JSONDecodeError:
        print("Warning: Could not parse JSON for relationships, returning raw.")
        return [{"llm_raw_response": response}]
    except Exception as e:
         print(f"Error parsing relationships: {e}")
         return [{"error": str(e), "llm_raw_response": response}]

# get_skip_patterns can be simpler - maybe move to repo_intel_tools?
# Let's keep it here for now as file processing might reveal more candidates.
async def suggest_skip_patterns(files: List[str], llm: LLMService) -> List[str]:
    """ Uses LLM to suggest additional directories/files to skip based on patterns seen. """
    print("[File Proc Tool] Suggesting skip patterns...")
    files_sample = files[:MAX_FILES_FOR_PROMPT]
    prompt = (
        f"Based on this file list:\n{json.dumps(files_sample, indent=2)}\n\n"
        f"Are there any other common patterns for directories or file types that should typically be excluded from analysis "
        f"(e.g., build artifacts, logs, compiled code, large assets)? List potential patterns (like '*.log', 'temp/')."
    )
    response = await llm.aask(prompt)
    # Basic parsing
    try:
        patterns = [p.strip() for p in response.split('\n') if p.strip() and not p.strip().startswith('{')]
        return patterns
    except Exception:
        return [response]
