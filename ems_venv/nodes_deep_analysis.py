from typing import Dict, Any
from file_proc_agent import FileProcessingAgent
from repo_intel_agent import RepoIntelAgent
from code_discovery_agent import CodeDiscoveryAgent
from web_search_agent import WebSearchAgent

# Initialize agents once for reuse
file_agent = FileProcessingAgent()
repo_agent = RepoIntelAgent()
code_agent = CodeDiscoveryAgent()
web_agent = WebSearchAgent()

async def repo_intelligence_node(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    return await repo_agent.process(state, config)

async def file_processing_node(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    # Restrict files based on include/exclude from config
    if "include_files" in config and config["include_files"]:
        state['files_to_process'] = config["include_files"]
    if "exclude_files" in config and config["exclude_files"]:
        files_set = set(state.get('files_to_process', []))
        files_set -= set(config["exclude_files"])
        state['files_to_process'] = list(files_set)
    state['analysis_depth'] = config.get("analysis_depth", "quick")
    return await file_agent.process(state, config)

async def code_discovery_node(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    state['verbosity'] = config.get("verbosity", "standard")
    return await code_agent.process(state, config)

async def web_search_node(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    if not config.get("enable_web_search", False):
        # Skip node, return state unchanged
        return state
    return await web_agent.process(state, config)
