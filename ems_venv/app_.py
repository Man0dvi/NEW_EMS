let's restart - give the following agent files :

1. analyses files by looping over a folder structure, skips unallowed list of files. and provides all of this context to llm which could be queried by user. this takes config files as input as instructions

3- websearch agent - if allowed by user

2- documentation agent- gets state from from previous two agents, creates a doc, exported as pdf

4- diagram agent - use some tool to create visualization using context

Here are the four agent files you requested, implementing the functionality described:

***

### 1. FileAnalysisAgent: analyzes files in folder structure, skips unallowed files, builds context

```python
import os
import asyncio
from typing import Dict, Any, List

from multi_agent_docs.services.llm_service import LLMService
from multi_agent_docs.tools.file_proc_tools import read_file_content, extract_code_elements

class FileAnalysisAgent:
    def __init__(self, llm_api_key: str = None):
        self.llm = LLMService(api_key=llm_api_key)

    async def process(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        project_path = state.get("project_path")
        include_files: List[str] = config.get("include_files", [])
        exclude_files: List[str] = config.get("exclude_files", [])
        analysis_depth = config.get("analysis_depth", "light")

        analyzed_chunks = []

        # Walk project path to find files, filter with include/exclude list
        for root, _, files in os.walk(project_path):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), project_path)
                if include_files and rel_path not in include_files:
                    continue
                if rel_path in exclude_files:
                    continue

                content = read_file_content(project_path, rel_path)
                if not content:
                    continue

                if analysis_depth == "deep":
                    chunks = await extract_code_elements(rel_path, content, self.llm)
                else:
                    chunks = [{"file": rel_path, "content": content}]

                analyzed_chunks.extend(chunks)

        state["file_analysis_chunks"] = analyzed_chunks
        return state
```

***

### 2. WebSearchAgent: conditionally runs DuckDuckGo web searches

```python
from typing import Dict, Any
from multi_agent_docs.services.duckduckgo_service import DuckDuckGoService

class WebSearchAgent:
    def __init__(self):
        self.duckduckgo = DuckDuckGoService()

    async def process(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        if not config.get("enable_web_search"):
            return state

        query = config.get("web_search_query")
        if query:
            results = await self.duckduckgo.search(query)
            state["web_search_results"] = results

        return state
```

***

### 3. DocumentationAgent: generates documentation PDF from state of previous agents

```python
from typing import Dict, Any
from multi_agent_docs.services.llm_service import LLMService
from multi_agent_docs.tools.documentation_tool import generate_pdf_report

class DocumentationAgent:
    def __init__(self, llm_api_key: str = None):
        self.llm = LLMService(api_key=llm_api_key)

    async def process(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        # Collect all relevant context from previous state
        file_chunks = state.get("file_analysis_chunks", [])
        web_search_results = state.get("web_search_results", [])

        # Generate a textual summary using LLM if needed, else use raw data
        prompt = self._build_prompt(file_chunks, web_search_results, config)
        summary = await self.llm.aask(prompt)

        # Generate PDF report file path
        pdf_path = generate_pdf_report(summary, config)

        state["documentation_pdf"] = pdf_path
        return state

    def _build_prompt(self, file_chunks, web_results, config):
        # Construct a prompt summarizing the context for LLM to generate docs
        content = ""
        content += f"Code Analysis Data: {len(file_chunks)} chunks.\n"
        content += f"Web Search Results: {len(web_results)} entries.\n"
        content += "Please generate the documentation as per user settings.\n"
        return content
```

***

### 4. DiagramAgent: makes visualizations from the current analysis state using Mermaid.js

```python
from typing import Dict, Any
from multi_agent_docs.tools.diagram_tool import create_mermaid_diagram

class DiagramAgent:
    def __init__(self):
        pass

    async def process(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        # Derive visualization input from current state
        file_chunks = state.get("file_analysis_chunks", [])

        # Generate mermaid syntax diagram string
        diagram_syntax = create_mermaid_diagram(file_chunks, config)

        state["mermaid_diagram"] = diagram_syntax
        return state
```

***

### Summary

- **FileAnalysisAgent** loops over project files as per configured include/exclude, runs chunk extraction & adds to state.
- **WebSearchAgent** optionally runs DuckDuckGo search by config, stores results.
- **DocumentationAgent** produces textual summary + generates PDF report based on the combined state.
- **DiagramAgent** produces Mermaid.js diagrams from file content context for visualization.

All agents accept and update the same shared state dictionary, forming progressive knowledge to support interactive querying.

***

If you want, I can now help generate:

- Supportive tool files mentioned here (`documentation_tool.py`, `diagram_tool.py`).
- The API routes to run and control these agents.
- Or orchestrating graph and nodes files integrating these agents.

Please specify next step.

