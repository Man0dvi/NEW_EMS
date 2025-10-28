# --- graph/graph.py ---

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
import operator

# Import the async node functions
from graph.nodes import (
    repo_intelligence_node,
    file_processing_node,
    code_discovery_node
)

# --- Define the State ---
# This dictionary structure will be passed between nodes
class CodeAnalysisState(TypedDict):
    project_path: str                 # Input: Path to the repository
    personas: List[str]               # Input: List of personas (e.g., ["SDE", "PM"])
    repo_files: List[str]             # Output of repo_intel
    tech_stack: List[str]             # Output of repo_intel
    structure_summary: Dict[str, Any] # Output of repo_intel
    architecture_summary: str         # Output of repo_intel
    complexity_assessment: str        # Output of repo_intel
    file_contents: Dict[str, str]     # Output of file_proc (optional)
    code_elements: List[Dict[str, Any]]# Output of file_proc
    dependencies: List[Dict[str, Any]]# Output of file_proc
    file_relationships: List[Dict[str, Any]]# Output of file_proc
    suggested_skip_patterns: List[str]# Output of file_proc
    semantic_chunks: List[Dict[str, Any]]# Output of code_discovery
    # Error fields (optional)
    repo_intel_error: str | None
    file_proc_error: str | None
    code_discovery_error: str | None

def build_analysis_graph() -> StateGraph:
    """ Builds the LangGraph analysis workflow. """
    print("[Graph] Building analysis graph...")
    # Initialize the StateGraph with our defined state
    graph = StateGraph(CodeAnalysisState)

    # Add nodes - Use the async functions from nodes.py
    graph.add_node("repo_intelligence", repo_intelligence_node)
    graph.add_node("file_processing", file_processing_node)
    graph.add_node("code_discovery", code_discovery_node)

    # Define the execution flow (edges)
    graph.set_entry_point("repo_intelligence")
    graph.add_edge("repo_intelligence", "file_processing")
    graph.add_edge("file_processing", "code_discovery")
    graph.add_edge("code_discovery", END) # End after code discovery

    # Compile the graph
    compiled_graph = graph.compile()
    print("[Graph] Analysis graph compiled.")
    try:
        print("\nGraph Structure:")
        compiled_graph.get_graph().print_ascii()
    except Exception as e:
        print(f"Could not print graph structure: {e}")
    return compiled_graph

# --- Example Invocation ---
if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv

    load_dotenv() # Load OPENAI_API_KEY etc.

    # --- IMPORTANT: Replace with the ACTUAL path to a code repository ---
    # --- Make it a small repo for initial testing ---
    TEST_REPO_PATH = "./test_repo" # Replace this! Create a small test repo folder.
    if not os.path.isdir(TEST_REPO_PATH):
         print(f"ERROR: Test repository path '{TEST_REPO_PATH}' not found.")
         print("Please create a directory with some sample code files or update the path.")
         exit()

    # Create the graph
    analysis_graph = build_analysis_graph()

    # Define initial state
    initial_state: CodeAnalysisState = {
        "project_path": TEST_REPO_PATH,
        "personas": ["SDE", "PM"],
        # Initialize other fields as None or empty where appropriate
        "repo_files": [],
        "tech_stack": [],
        "structure_summary": {},
        "architecture_summary": "",
        "complexity_assessment": "",
        "file_contents": {},
        "code_elements": [],
        "dependencies": [],
        "file_relationships": [],
        "suggested_skip_patterns": [],
        "semantic_chunks": [],
        "repo_intel_error": None,
        "file_proc_error": None,
        "code_discovery_error": None
    }

    # Asynchronous invocation
    async def run_graph():
        print(f"\n--- Running analysis on: {TEST_REPO_PATH} ---")
        # Use astream_events for detailed logs or ainvoke for final result
        config = {"recursion_limit": 10} # Add recursion limit
        final_state = await analysis_graph.ainvoke(initial_state, config=config)
        print("\n--- Graph Execution Complete ---")

        # Pretty print the final state
        print("\nFinal State:")
        # Use json dumps for better formatting of complex structures
        print(json.dumps(final_state, indent=2, default=str)) # Use default=str for non-serializable

    # Run the async function
    try:
        asyncio.run(run_graph())
    except Exception as e:
        print(f"\n--- An error occurred during graph execution ---")
        print(e)
        import traceback
        traceback.print_exc()
