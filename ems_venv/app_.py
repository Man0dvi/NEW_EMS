# --- graph/nodes.py ---

import os
from typing import Dict, Any
# Import agents
from agents.repo_intel_agent import RepoIntelligenceAgent
from agents.file_proc_agent import FileProcessingAgent
from agents.code_discovery_agent import CodeDiscoveryAgent

# Instantiate agents once (or instantiate LLMService once and pass)
# For simplicity here, instantiate per call, but sharing LLMService is better
LLM_API_KEY = os.getenv("OPENAI_API_KEY") # Ensure API key is available

# --- Node functions MUST be async to call async agent methods ---

async def repo_intelligence_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """ Node that runs the RepoIntelligenceAgent. """
    print("\n>>> Executing Repo Intelligence Node <<<")
    agent = RepoIntelligenceAgent(llm_api_key=LLM_API_KEY)
    # Use await to call the async process method
    result_state = await agent.process(state)
    print("<<< Finished Repo Intelligence Node >>>")
    # Return only the modified parts of the state expected by LangGraph
    # Or, if returning the whole state, ensure keys match StateGraph definition
    return result_state # Agent modifies state directly

async def file_processing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """ Node that runs the FileProcessingAgent. """
    print("\n>>> Executing File Processing Node <<<")
    agent = FileProcessingAgent(llm_api_key=LLM_API_KEY)
    result_state = await agent.process(state)
    print("<<< Finished File Processing Node >>>")
    return result_state

async def code_discovery_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """ Node that runs the CodeDiscoveryAgent. """
    print("\n>>> Executing Code Discovery Node <<<")
    agent = CodeDiscoveryAgent(llm_api_key=LLM_API_KEY)
    result_state = await agent.process(state)
    print("<<< Finished Code Discovery Node >>>")
    return result_state
