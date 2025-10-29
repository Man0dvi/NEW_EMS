now give the agent files

Here are the two agent files you requested: `file_analysis_agent.py` and `web_search_agent.py`, designed to integrate with the graph and nodes for your configurable multi-agent orchestration.

***

### file_analysis_agent.py

```python
import asyncio
import logging
from typing import Dict, Any, List
from multi_agent_docs.services.llm_service import LLMService
from multi_agent_docs.tools.file_proc_tools import read_file_content, extract_code_elements
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class FileAnalysisAgent:
    def __init__(self, llm_api_key: str = None):
        self.llm = LLMService(api_key=llm_api_key)

    async def process(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        project_path = state.get("project_path")
        files_to_analyze: List[str] = config.get("include_files", [])
        exclude_files: List[str] = config.get("exclude_files", [])
        analysis_depth = config.get("analysis_depth", "light")

        logger.info("FileAnalysisAgent starting with %d files and depth %s", len(files_to_analyze), analysis_depth)

        analyzed_chunks = []

        for file_path in files_to_analyze:
            if file_path in exclude_files:
                logger.info("Skipping excluded file: %s", file_path)
                continue

            content = read_file_content(project_path, file_path)
            if not content:
                logger.warning("No content found for %s", file_path)
                continue

            if analysis_depth == "deep":
                chunks = await extract_code_elements(file_path, content, self.llm)
            else:
                chunks = [{"file": file_path, "content": content}]

            analyzed_chunks.extend(chunks)

        state["file_analysis_chunks"] = analyzed_chunks
        logger.info("FileAnalysisAgent completed with %d chunks", len(analyzed_chunks))
        return state
```

***

### web_search_agent.py

```python
import logging
from typing import Dict, Any
from multi_agent_docs.services.duckduckgo_service import DuckDuckGoService

logger = logging.getLogger(__name__)

class WebSearchAgent:
    def __init__(self):
        self.duckduckgo = DuckDuckGoService()

    async def process(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        query = config.get("web_search_query")
        if not query:
            logger.info("No web search query provided, skipping WebSearchAgent.")
            return state

        logger.info("WebSearchAgent searching for: %s", query)
        results = await self.duckduckgo.search(query)
        state["web_search_results"] = results
        logger.info("WebSearchAgent returned %d results", len(results))
        return state
```

***

These agent files assume you have existing or upcoming implementations for:

- `read_file_content` and `extract_code_elements` in `file_proc_tools`.
- An async DuckDuckGo search service `DuckDuckGoService` with an async `search(query)` function.
- `LLMService` wrapping your OpenAI calls.

They fit into the graph orchestration as nodes to be run in sequence or parallel depending on config flags.

***

Would you like next to create:

- The DuckDuckGo search service?
- The nodes file connecting these agents as graph nodes?
- Or the orchestration graph logic itself?

