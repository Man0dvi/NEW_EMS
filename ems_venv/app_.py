this is the previous graph I had, use the same to make the node_deep and graph_deep

Here is how you would structure `node_deep.py` and `graph_deep.py` so they fit with the style and sequence of your previous implementation, **using the explicit function-based node and edge setup modeled after your screenshots**.

***

## node_deep.py

```python
from file_analysis_agent import FileAnalysisAgent
from web_search_agent import WebSearchAgent
from documentation_agent import DocumentationAgent
from diagram_agent import DiagramAgent

# Each agent node gets its async callable function

async def file_analysis_node(state, config):
    agent = FileAnalysisAgent()
    return await agent.process(state, config)

async def web_search_node(state, config):
    agent = WebSearchAgent()
    return await agent.process(state, config)

async def documentation_node(state, config):
    agent = DocumentationAgent()
    return await agent.process(state, config)

async def diagram_node(state, config):
    agent = DiagramAgent()
    return await agent.process(state, config)
```

***

## graph_deep.py

This respects your "graph-building" flow, using LangGraph's or a similar internal graph-style API:

```python
from langgraph import StateGraph  # or however you import your graph builder!
from node_deep import (
    file_analysis_node,
    web_search_node,
    documentation_node,
    diagram_node,
)

def build_deep_analysis_graph(CodeAnalysisState):
    graph_builder = StateGraph(CodeAnalysisState)

    # Add nodes
    graph_builder.add_node("file_analysis", file_analysis_node)
    graph_builder.add_node("web_search", web_search_node)
    graph_builder.add_node("documentation", documentation_node)
    graph_builder.add_node("diagram", diagram_node)

    # Sequential and conditional edges
    graph_builder.set_entry_point("file_analysis")
    graph_builder.add_edge("file_analysis", "web_search")       # Always runs after file_analysis
    graph_builder.add_edge("web_search", "documentation")       # Always runs after web_search
    graph_builder.add_edge("documentation", "diagram")          # Always runs after docs
    graph_builder.add_edge("diagram", "END")                    # Graph ends after diagram

    # You can make web_search conditional by overriding the node to no-op if not enabled in config,
    # or dynamically removing the edge when building the graph—see below alternate for conditional disabling:

    # if not config.get("enable_web_search"):
    #     graph_builder.add_edge("file_analysis", "documentation")
    # else:
    #     graph_builder.add_edge("file_analysis", "web_search")
    #     graph_builder.add_edge("web_search", "documentation")
    # graph_builder.add_edge("documentation", "diagram")
    # graph_builder.add_edge("diagram", "END")

    # Return compiled graph
    return graph_builder.compile()
```

***

### Notes:
- Your previous approach caches the compiled graph, so you can still do that outside this function as in your past code.
- Each node is a simple async function wrapping the respective agent's process method.
- Conditional disabling of web search can be handled either by making `web_search_node` a no-op (instantly returns state) if not enabled, or by removing the edge and jumping straight to documentation.

***

This refactor closely matches your previous, working pattern with extensibility for new agent/nodes!

Would you like further refinements or guidance on the main orchestration/calling flow or API integration next?

Citations:
[1] selected_image_1183273465671543608.jpg https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/13630366/4f9cc58a-ce3e-4171-b018-ac842d399cd1/selected_image_1183273465671543608.jpg?AWSAccessKeyId=ASIA2F3EMEYE7EC6ORVR&Signature=4nAfb1UIsA%2By%2BWAg3nVGHiedECI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEBoaCXVzLWVhc3QtMSJHMEUCIFTkZp%2F1x1FkIwLBswKjjA1O5KNFhqbTUJQE3yrNapmIAiEAvM7bInJgqqlsZN8qwcVKeZbI91Yc%2FFpjOK2qmkmPTh8q%2FAQI0%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDITGQ8lDd5gZ6i75tirQBMfy1CLJvPZeZmbMxo31a3RR8e8LP499dPi2Wl82%2BGdgdYDgNxKdjIDcfaw39R3sPKowrGPRU812F3XAhWAQQt8D3LhQTWzIhStNS4BC%2F1yE46wKo2lYIuvz4H1I1E3YQqfebwhGMGr1Qbq3QOD9qdttTt67YdAIPCjyZkkQiOpU4K3sgyg0%2BB0B2BJV5Rjw8O%2FDSR1dhtEypGS%2BTDwaPKwxxM%2Fnfhgzu%2FIk3HzfKf7TigE3s1kFt0tHGiLr0bB09Ey68kQDPWViKmj%2FTzhBYIuB7GYw2gS1KDLrU30Chlw7z7h8C%2FsEhSkKGuwP9%2BtF3ju9K0uLpWQKhi3viVJsN5q2pklyw2tUK7WteXWTjCxUyJkl4uPbpA8P8oXrIurNoJRNto9ub6o21gXorvms8ugeNO8LA1PhPsw0cXNdtWGw%2B0gZBX36aGs6E5bbZdR60jhzGdOsyTWJv7OVCD42eVXNt1KHgco8lIlwQsmHBwcBKYfYpcqLuhLZ1%2BCH7qH3FouTNMZHo3RfPkTORzzdJ9yZDqh1ExBillp7MHd2w6AUVg8GPmaJgB%2BdUWgfEEGY13G6pRZWpIzDTUQ4TqDZge04WxsXe4rMJV6UpnJbK7HNN4SnWd9ftI3SDNpzgsWSHU8x4Lzok%2Fvr9UPS5kThiy5KQn531jDCWyVWjDkYzVnRG9KNzxHP61FFJz81i%2B4XJ%2Fbl%2B5kuAV23z707Q7CXFaqqx%2Buj9gdnAWB9FzvUpkg%2FR6PTGS5Nq9HVkHjrjbwF%2BSPLFT%2Fko4Uk4w6%2BzGGHFNkwhrqHyAY6mAFbqhpCGyl1H%2BpLqe8VOl6LoN6N9fZnwonhuZgnU%2FiReio7HGEfbPO23aDNm0ihNSdxG9mD9gxrWaXu%2FzsBQudPmpccxRKKi2LAHZwH5AgJDpyi33hAYtFMEPB1LfsJ6LtK0eQmgbifKh4DH7ii4F%2BoRrFEWw8ejn%2B3Z8HBeZJhdHmzKgtyDcjn4uHocZfas%2FDM937RW040eA%3D%3D&Expires=1761732709
[2] selected_image_6213899448013275800.jpg https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/13630366/b60230c6-0e05-4434-a195-fca1632eff9b/selected_image_6213899448013275800.jpg?AWSAccessKeyId=ASIA2F3EMEYE7EC6ORVR&Signature=iZyDHz1x5%2FTBZbumQN%2FgulQLkjI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEBoaCXVzLWVhc3QtMSJHMEUCIFTkZp%2F1x1FkIwLBswKjjA1O5KNFhqbTUJQE3yrNapmIAiEAvM7bInJgqqlsZN8qwcVKeZbI91Yc%2FFpjOK2qmkmPTh8q%2FAQI0%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDITGQ8lDd5gZ6i75tirQBMfy1CLJvPZeZmbMxo31a3RR8e8LP499dPi2Wl82%2BGdgdYDgNxKdjIDcfaw39R3sPKowrGPRU812F3XAhWAQQt8D3LhQTWzIhStNS4BC%2F1yE46wKo2lYIuvz4H1I1E3YQqfebwhGMGr1Qbq3QOD9qdttTt67YdAIPCjyZkkQiOpU4K3sgyg0%2BB0B2BJV5Rjw8O%2FDSR1dhtEypGS%2BTDwaPKwxxM%2Fnfhgzu%2FIk3HzfKf7TigE3s1kFt0tHGiLr0bB09Ey68kQDPWViKmj%2FTzhBYIuB7GYw2gS1KDLrU30Chlw7z7h8C%2FsEhSkKGuwP9%2BtF3ju9K0uLpWQKhi3viVJsN5q2pklyw2tUK7WteXWTjCxUyJkl4uPbpA8P8oXrIurNoJRNto9ub6o21gXorvms8ugeNO8LA1PhPsw0cXNdtWGw%2B0gZBX36aGs6E5bbZdR60jhzGdOsyTWJv7OVCD42eVXNt1KHgco8lIlwQsmHBwcBKYfYpcqLuhLZ1%2BCH7qH3FouTNMZHo3RfPkTORzzdJ9yZDqh1ExBillp7MHd2w6AUVg8GPmaJgB%2BdUWgfEEGY13G6pRZWpIzDTUQ4TqDZge04WxsXe4rMJV6UpnJbK7HNN4SnWd9ftI3SDNpzgsWSHU8x4Lzok%2Fvr9UPS5kThiy5KQn531jDCWyVWjDkYzVnRG9KNzxHP61FFJz81i%2B4XJ%2Fbl%2B5kuAV23z707Q7CXFaqqx%2Buj9gdnAWB9FzvUpkg%2FR6PTGS5Nq9HVkHjrjbwF%2BSPLFT%2Fko4Uk4w6%2BzGGHFNkwhrqHyAY6mAFbqhpCGyl1H%2BpLqe8VOl6LoN6N9fZnwonhuZgnU%2FiReio7HGEfbPO23aDNm0ihNSdxG9mD9gxrWaXu%2FzsBQudPmpccxRKKi2LAHZwH5AgJDpyi33hAYtFMEPB1LfsJ6LtK0eQmgbifKh4DH7ii4F%2BoRrFEWw8ejn%2B3Z8HBeZJhdHmzKgtyDcjn4uHocZfas%2FDM937RW040eA%3D%3D&Expires=1761732709
