from langgraph import StateGraph
from model import CodeAnalysisState
from nodes_deep_analysis import (
    repo_intelligence_node,
    file_processing_node,
    code_discovery_node,
    web_search_node,
)

def build_deep_analysis_graph() -> StateGraph:
    graph = StateGraph(CodeAnalysisState)

    graph.add_node("repo_intelligence", repo_intelligence_node)
    graph.add_node("file_processing", file_processing_node)
    graph.add_node("code_discovery", code_discovery_node)
    graph.add_node("web_search", web_search_node)

    # entry point
    graph.set_entry_point("repo_intelligence")

    # Edges (web search runs conditionally after code discovery)
    graph.add_edge("repo_intelligence", "file_processing")
    graph.add_edge("file_processing", "code_discovery")
    graph.add_edge("code_discovery", "web_search")
    graph.add_edge("web_search", "END")

    return graph
