# --- graph_builder.py (Refactored for Single Server) ---

import os
import operator
import json
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Literal
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

# Import the UPDATED agent runner tools
from agent_runners import get_agent_runner_tools

# Import MCP client
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# --- 1. Define Agent State (Unchanged) ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    original_query: str
    sub_tasks: List[str]
    completed_tasks: Annotated[List[str], operator.add]
    task_results: Dict[str, List[Dict[str, Any]]]

# --- 2. Setup MCP Client for Agent 1's Internal Tools (Use Consolidated Client Config) ---
SERVER_BASE_URL = os.getenv("ALL_TOOLS_URL", "http://localhost:8000")
try:
    # This client connects ONLY to the Coordinator's namespace on the single server
    agent_1_client = MultiServerMCPClient({
        "Coordinator_Tools": { # Namespace must match FastMCP("Coordinator_Tools")
            "transport": "streamable_http",
            "url": f"{SERVER_BASE_URL}/coordinator/mcp" # Point to coordinator's mount path
        }
    })
    print("[Graph Builder] Configured client for Agent 1 Tools (on consolidated server).")
except Exception as e:
    print(f"[Graph Builder] ERROR configuring Agent 1 client: {e}")
    agent_1_client = None

# --- 3. Function to Fetch All Tools (Fetch internal tools + get runners) ---
_coordinator_tools_cache = None
async def get_coordinator_tools() -> List[BaseTool]:
    """Fetches Agent 1's internal tools and combines with agent runners."""
    global _coordinator_tools_cache
    if _coordinator_tools_cache: return _coordinator_tools_cache

    internal_tools = []
    if agent_1_client:
        try:
            # Fetch tools ONLY from the Coordinator_Tools namespace
            internal_tools = await agent_1_client.get_tools(namespace="Coordinator_Tools")
            print(f"[Graph Builder] Fetched {len(internal_tools)} internal tools for Agent 1.")
        except Exception as e:
            print(f"[Graph Builder] ERROR fetching Agent 1 internal tools: {e}")

    # Combine internal tools with the imported agent runner tools
    all_tools = internal_tools + get_agent_runner_tools()
    print(f"[Graph Builder] Total tools available to coordinator: {[tool.name for tool in all_tools]}")
    if not all_tools: print("[Graph Builder] WARNING: No tools loaded!")
    _coordinator_tools_cache = all_tools
    return all_tools

# --- 4. Define Graph Nodes (decompose_query_node, process_tool_results_node, final_synthesis_answer_node are largely unchanged, but use correct namespace) ---

async def decompose_query_node(state: AgentState):
    """Calls Agent 1's internal decomposition tool."""
    print("\n--- Running Node: Decompose Query ---")
    if not agent_1_client: raise ConnectionError("Agent 1 client not available.")
    user_message = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), state["messages"][-1] if state["messages"] else None)
    if not user_message: return {"messages": [AIMessage(content="Error: No query found.")]}
    original_query = user_message.content
    print(f"Original Query: {original_query}")
    try:
        # CALL WITH NAMESPACE
        sub_tasks = await agent_1_client.call("Coordinator_Tools.query_decomposition_tool", user_query=original_query)
        print(f"Decomposed Tasks: {sub_tasks}")
        if not isinstance(sub_tasks, list) or not all(isinstance(t, str) for t in sub_tasks): sub_tasks = [str(sub_tasks)]
        return {"original_query": original_query, "sub_tasks": sub_tasks, "completed_tasks": [], "task_results": {}}
    except Exception as e: return {"messages": [AIMessage(content=f"Error during decomposition: {e}")]}

def process_tool_results_node(state: AgentState):
    """Processes the output of the tool node and updates the state."""
    print("\n--- Running Node: Process Tool Results ---")
    # --- (Implementation is the same as before - relies on supervisor logic) ---
    messages = state["messages"]; last_message = messages[-1]
    if not isinstance(last_message, ToolMessage): return {}
    tool_call_id = last_message.tool_call_id; tool_content = last_message.content
    tool_name = "unknown_tool"; tool_args = {}
    for msg in reversed(messages[:-1]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc['id'] == tool_call_id: tool_name = tc.get('name', 'unknown'); tool_args = tc.get('args', {}); break
            if tool_name != "unknown_tool": break
    print(f"Processing result for tool: {tool_name}")
    sub_tasks = state.get("sub_tasks", []); completed_tasks = state.get("completed_tasks", [])
    pending_tasks = [t for t in sub_tasks if t not in completed_tasks]
    current_task = pending_tasks[0] if pending_tasks else "general_results"
    updated_task_results = state['task_results'].copy()
    current_results_list = updated_task_results.get(current_task, [])
    if not isinstance(current_results_list, list): current_results_list = [current_results_list]
    current_results_list.append({"tool_called": tool_name, "tool_args": tool_args, "tool_result": tool_content})
    updated_task_results[current_task] = current_results_list
    print(f"Stored result under task: {current_task}")
    # Let supervisor decide on completion, only return results
    return {"task_results": updated_task_results}
    # --- (End of process_tool_results_node) ---

def final_synthesis_answer_node(state: AgentState):
    """Extracts the final synthesis report from the last ToolMessage."""
    print("\n--- Running Node: Final Synthesis Answer ---")
    # --- (Implementation is the same as before) ---
    messages = state["messages"]; last_message = messages[-1]
    if isinstance(last_message, ToolMessage):
        tool_call_id = last_message.tool_call_id
        for msg in reversed(messages[:-1]):
             if isinstance(msg, AIMessage) and msg.tool_calls:
                  if any(tc['id'] == tool_call_id and tc['name'] == 'result_synthesis_tool' for tc in msg.tool_calls):
                       print("Synthesis result found.")
                       return {"messages": [AIMessage(content=str(last_message.content))]} # Ensure content is string
                  break
    print("Warning: Final node reached without synthesis result.")
    last_ai = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
    return {"messages": [last_ai]} if last_ai else {}
    # --- (End of final_synthesis_answer_node) ---


# --- 5. Build and Compile the Graph (Supervisor and Edges are the same) ---
async def compile_graph():
    """Fetches tools and compiles the LangGraph workflow."""
    print("[Graph Builder] Compiling Coordinator Graph...")
    all_tools = await get_coordinator_tools()
    if not all_tools: return None

    supervisor_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    supervisor_llm_bound = supervisor_llm.bind_tools(all_tools)

    def supervisor_node_func(state: AgentState):
        """Supervisor LLM decides the next action."""
        print("\n--- Running Node: Supervisor ---")
        if "sub_tasks" not in state or state["sub_tasks"] is None: return {"messages": [AIMessage(content="Internal Error: Sub-tasks missing.")]}
        sub_tasks = state.get("sub_tasks", []); completed_tasks = state.get("completed_tasks", [])
        pending_tasks = [t for t in sub_tasks if t not in completed_tasks]; all_done = sub_tasks and not pending_tasks
        if all_done:
            print("All sub-tasks seem complete by count. Requesting final synthesis.")
            original_query = state["original_query"]; all_gathered_data = state["task_results"]
            synthesis_tool_call = {"name": "result_synthesis_tool", "args": {"original_query": original_query, "all_gathered_data": all_gathered_data}, "id": f"synthesis_call_{len(state['messages'])}"}
            return {"messages": [AIMessage(content="All tasks complete. Synthesizing final report.", tool_calls=[synthesis_tool_call])]}
        else:
            supervisor_messages = state['messages'].copy()
            next_task = pending_tasks[0] if pending_tasks else "None"
            # Slightly improved prompt for supervisor focus
            task_status_prompt = f"\n\nSystem Note: Goal='{state['original_query']}'. Sub-tasks={sub_tasks}. Completed={completed_tasks}. Focus on first pending task: '{next_task}'. Review history & tools, decide next action. If task '{next_task}' requires multiple tool calls, call the next logical one. If you believe '{next_task}' is complete based on last tool result, state reasoning clearly and focus on the *next* pending task. If all tasks are done, call 'result_synthesis_tool'."
            supervisor_messages.append(SystemMessage(content=task_status_prompt))
            print(f"Supervisor sending {len(supervisor_messages)} messages to LLM...")
            response = supervisor_llm_bound.invoke(supervisor_messages)
            print(f"Supervisor received response: {response.content[:100]}..., Tool Calls: {response.tool_calls}")
            # --- Mark task as complete ONLY if supervisor explicitly says so ---
            # --- This requires more complex parsing or a dedicated state update ---
            # --- For now, rely on task count for synthesis trigger ---
            # --- Add the response to messages ---
            return {"messages": [response]}

    tool_node = ToolNode(all_tools)
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("decompose", decompose_query_node)
    graph_builder.add_node("supervisor", supervisor_node_func)
    graph_builder.add_node("execute_tools", tool_node)
    graph_builder.add_node("process_results", process_tool_results_node)
    graph_builder.add_node("final_answer", final_synthesis_answer_node)

    graph_builder.set_entry_point("decompose")
    graph_builder.add_edge("decompose", "supervisor")
    graph_builder.add_conditional_edges("supervisor", tools_condition, {"tools": "execute_tools", END: END}) # If supervisor decides to end (e.g., error, or maybe implicit finish?)
    graph_builder.add_edge("execute_tools", "process_results")
    def after_processing_condition(state: AgentState) -> Literal["supervisor", "final_answer"]:
        """Check if the last tool executed was the synthesis tool."""
        print("\n--- Running Edge Logic: After Processing ---")
        # --- (Implementation is the same as before) ---
        last_message = state["messages"][-1]
        if isinstance(last_message, ToolMessage):
             tool_call_id = last_message.tool_call_id
             for msg in reversed(state["messages"][:-1]):
                  if isinstance(msg, AIMessage) and msg.tool_calls:
                       if any(tc['id'] == tool_call_id and tc['name'] == 'result_synthesis_tool' for tc in msg.tool_calls):
                            print("Decision: Synthesis tool ran. Go to Final Answer.")
                            return "final_answer"
                       break
        print("Decision: Synthesis not run. Loop back to Supervisor.")
        return "supervisor"
        # --- (End of condition logic) ---
    graph_builder.add_conditional_edges("process_results", after_processing_condition, {"supervisor": "supervisor", "final_answer": "final_answer"})
    graph_builder.add_edge("final_answer", END)

    coordinator_graph = graph_builder.compile()
    print("[Graph Builder] Coordinator graph compiled successfully.")
    try: print("\nGraph Structure:"); coordinator_graph.get_graph().print_ascii()
    except Exception as e: print(f"Could not print graph: {e}")
    return coordinator_graph

if __name__ == '__main__':
    async def test_graph_compilation(): graph = await compile_graph()
    print("Graph builder defined. Import 'compile_graph'.")
    import asyncio
    asyncio.run(test_graph_compilation())
