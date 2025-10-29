import asyncio
from typing import List, Dict, Any

async def extract_code_elements(file_path: str, content: str, llm) -> List[Dict[str, Any]]:
    prompt = f"Analyze this file in detail: {file_path}\n\n{content}\nReturn code elements and their roles as JSON."
    result = await llm.aask(prompt)
    import json
    try:
        return json.loads(result)
    except Exception:
        return []

async def extract_dependencies(file_path: str, content: str, llm) -> Dict[str, Any]:
    prompt = f"Analyze dependencies in this file: {file_path}\n\n{content}\nReturn dependencies as JSON."
    result = await llm.aask(prompt)
    import json
    try:
        return json.loads(result)
    except Exception:
        return {}

async def extract_file_relationships(files: List[str], file_contents: Dict[str, str], llm) -> List[Dict[str, Any]]:
    prompt = f"Analyze relationships between these files: {files}\n\nContents:\n"
    for f in files:
        prompt += f"\n{f}: {file_contents[f][:100]}..."  # snippet for brevity
    prompt += "\nReturn relationships as JSON."
    result = await llm.aask(prompt)
    import json
    try:
        return json.loads(result)
    except Exception:
        return []

async def suggest_skip_patterns(files: List[str], llm) -> List[str]:
    prompt = f"Suggest patterns for ignoring files or folders to improve analysis for files:\n{files}\n"
    result = await llm.aask(prompt)
    try:
        import json
        return json.loads(result)
    except Exception:
        return []
