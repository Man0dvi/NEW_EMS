from typing import Dict, Any
from multi_agent_docs.services.llm_service import LLMService
import json

class LLMFlowDiagramAgent:
    def __init__(self, llm_api_key: str = None):
        self.llm = LLMService(api_key=llm_api_key)

    async def process(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        final_state = state
        relationships = final_state.get("file_relationships") or final_state.get("dependencies") or []

        relationships_json = json.dumps(relationships, indent=2)
        prompt = (
            "Given the following JSON array of file-to-file relationships or dependencies in a codebase, "
            "generate a Mermaid flowchart (graph TD) that visualizes these relationships. "
            "Each relationship contains source, target and optionally relation type. "
            "Example output format:\n\n"
            "graph TD\n  A --> B\n  B --> C\n  A -- calls --> C\n\n"
            "JSON Data:\n"
            f"{relationships_json}\n\n"
            "Mermaid graph TD:"
        )

        diagram = await self.llm.aask(prompt)
        state["mermaid_diagram"] = diagram.strip()
        return state
