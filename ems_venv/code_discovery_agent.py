# code_discovery_agent.py

from typing import Dict, Any, List, Optional
from code_discovery_tools import summarize_technical_details, summarize_business_logic

class CodeDiscoveryAgent:
    async def process(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        verbosity: str = config.get("verbosity", "standard")
        personas: List[str] = config.get("personas", [])

        # Extract files to analyze from state or fallback
        files_to_process: List[str] = state.get("files_to_process", [])
        if not files_to_process:
            files_to_process = state.get("all_files", [])

        # Container for results per persona
        sde_analysis_results: List[str] = []
        pm_summary_results: List[str] = []

        # Loop over files and execute persona-specific summarization
        for filename in files_to_process:
            file_content = state.get("file_contents", {}).get(filename, "")
            if not file_content:
                continue  # Skip if no content in this example

            if "SDE" in personas:
                # More in-depth technical summary for dev persona
                result = await summarize_technical_details(file_content, verbosity)
                sde_analysis_results.append(f"File: {filename}\n" + result)

            if "PM" in personas:
                # Business focused summaries for product manager persona
                result = await summarize_business_logic(file_content, verbosity)
                pm_summary_results.append(f"File: {filename}\n" + result)

        # Update state with persona specific outputs
        if sde_analysis_results:
            state["sde_analysis"] = "\n\n".join(sde_analysis_results)
        if pm_summary_results:
            state["pm_summary"] = "\n\n".join(pm_summary_results)

        # Add any additional summary or aggregate
        return state
