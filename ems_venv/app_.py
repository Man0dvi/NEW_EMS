okay then let's create two files at a time

Let's begin by creating the two key files for the orchestration, as you requested:

1. `graph.py`: the analysis graph orchestration which will conditionally run nodes, manage the overall flow, handle pause/resume, and maintain the final state object.

2. `nodes.py`: which will define the individual nodes (agents) such as `FileAnalysisNode` and `WebSearchNode` that perform their specific tasks and update the shared state.

***

### 1. graph.py

```python
import asyncio
from typing import Dict, Any, Optional

from nodes import FileAnalysisNode, WebSearchNode

class AnalysisGraph:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state: Dict[str, Any] = {}  # This holds the final state accumulated
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Initially not paused

        # Initialize nodes based on config
        self.file_analysis_node = FileAnalysisNode()
        self.web_search_node = WebSearchNode() if config.get("enable_web_search") else None

    async def run(self):
        self._paused = False
        self._pause_event.set()

        try:
            # Run file analysis node
            await self._wait_if_paused()
            self.state = await self.file_analysis_node.run(self.state, self.config)

            # Conditionally run web search node
            if self.web_search_node:
                await self._wait_if_paused()
                self.state = await self.web_search_node.run(self.state, self.config)

        except asyncio.CancelledError:
            # Graceful cancellation pausing the graph
            self._paused = True
            self._pause_event.clear()
            raise

        return self.state

    async def _wait_if_paused(self):
        # Wait if paused
        await self._pause_event.wait()

    def pause(self):
        # Clear the event to pause graph run
        self._paused = True
        self._pause_event.clear()

    def resume(self):
        # Set the event to resume graph run
        self._paused = False
        self._pause_event.set()

    def is_paused(self) -> bool:
        return self._paused
```

***

### 2. nodes.py

```python
from typing import Dict, Any

from agents.file_analysis_agent import FileAnalysisAgent
from agents.web_search_agent import WebSearchAgent

class FileAnalysisNode:
    def __init__(self):
        self.agent = FileAnalysisAgent()

    async def run(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        # Run file by file analysis, respecting config options
        state = await self.agent.process(state, config)
        return state

class WebSearchNode:
    def __init__(self):
        self.agent = WebSearchAgent()

    async def run(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        # Run web search tool only if enabled by config
        if not config.get("enable_web_search", False):
            return state

        state = await self.agent.process(state, config)
        return state
```

***

### Next Steps

- Implement `FileAnalysisAgent` and `WebSearchAgent` classes in your agents directory (reuse existing logic and integrate config).
- Update your API endpoints to create an `AnalysisGraph` with user config, run the graph asynchronously.
- Implement API endpoints to pause/resume by calling graph's `pause()` and `resume()`.
- Store and return incremental `state` from the graph for UI updates and search queries.

***

Would you like me to create the first agent file next (e.g. `file_analysis_agent.py`), or help write the related FastAPI endpoints for this orchestration?

