# --- graph_builder.py (Refactored with create_agent Supervisor) ---

import os
import operator
import json
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Literal, Optional
from dotenv import load_dotenv

# Core LangGraph imports
from langgraph.graph import StateGraph, END
# ToolNode is no longer needed directly in the graph structure
# tools_condition is replaced by checking agent output

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent # Use the standard agent creation function
from langchain.tools import BaseTool

# Local imports
from agent_runners import get_agent_runner_tools # Tool wrappers for Agents 2,3,4
from langchain_mcp_adapters.client import MultiServerMCPClient # MCP client

load_dotenv()

# --- 1. Define Agent State (Unchanged) ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    original_query: str
    sub_tasks: List[str]
    completed_tasks: Annotated[List[str], operator.add]
    task_results: Dict[str, List[Dict[str, Any]]]

# --- 2. Setup MCP Client for Agent 1's Internal Tools (Unchanged) ---
SERVER_URL = os.getenv("ALL_TOOLS_URL", "http://localhost:8000/mcp")
try:
    agent_1_client = MultiServerMCPClient({
        "Coordinator_Tools": {"transport": "streamable_http", "url": SERVER_URL}
    })
    print(f"[Graph Builder] Configured client for Agent 1 Tools (on {SERVER_URL}).")
except Exception as e:
    print(f"[Graph Builder] ERROR configuring Agent 1 client: {e}")
    agent_1_client = None

# --- 3. Function to Fetch All Tools (Unchanged logic, includes cache) ---
_coordinator_tools_cache: Optional[List[BaseTool]] = None
_internal_tools_cache: Optional[List[BaseTool]] = None
_tool_cache_lock = asyncio.Lock()

async def get_internal_coordinator_tools() -> List[BaseTool]:
    """ Fetches and caches *only* Agent 1's internal tools. """
    global _internal_tools_cache
    async with _tool_cache_lock:
        # ... (implementation unchanged) ...
        if _internal_tools_cache is not None: return _internal_tools_cache
        if not agent_1_client: _internal_tools_cache = []; return []
        try: tools = await agent_1_client.get_tools(namespace="Coordinator_Tools"); print(f"Fetched {len(tools)} internal tools."); _internal_tools_cache = tools; return tools
        except Exception as e: print(f"ERROR fetching internal tools: {e}"); _internal_tools_cache = []; return []

async def find_internal_tool(tool_name: str) -> Optional[BaseTool]:
    """ Finds a specific internal tool object. """
    internal_tools = await get_internal_coordinator_tools()
    for t in internal_tools:
        if t.name == tool_name: return t
    return None

async def get_all_coordinator_tools() -> List[BaseTool]:
    """ Fetches internal tools and combines with runner tools. """
    global _coordinator_tools_cache
    async with _tool_cache_lock:
        # ... (implementation unchanged) ...
        if _coordinator_tools_cache is not None: return _coordinator_tools_cache
        internal_tools = await get_internal_coordinator_tools()
        runner_tools = get_agent_runner_tools()
        all_tools = internal_tools + runner_tools
        print(f"Total tools: {[t.name for t in all_tools]}")
        _coordinator_tools_cache = all_tools; return all_tools

# --- 4. Define Graph Nodes ---

async def decompose_query_node(state: AgentState) -> Dict[str, Any]:
    """ Calls Agent 1's internal decomposition tool using get_tools + tool.ainvoke. """
    print("\n--- Node: Decompose Query ---")
    user_message = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), state["messages"][-1] if state["messages"] else None)
    if not user_message: return {"messages": [AIMessage(content="Error: No query.")]}
    original_query = user_message.content; print(f"Query: {original_query}")
    decomp_tool = await find_internal_tool("query_decomposition_tool")
    if not decomp_tool: return {"messages": [AIMessage(content="Internal Error: Decomposition tool missing.")]}
    try:
        print("Invoking decomposition tool...")
        sub_tasks = await decomp_tool.ainvoke({"user_query": original_query})
        print(f"Tasks: {sub_tasks}")
        if not isinstance(sub_tasks, list) or not all(isinstance(t, str) for t in sub_tasks): sub_tasks = [str(sub_tasks)]
        # Return updates to state
        return {
            "original_query": original_query,
            "sub_tasks": sub_tasks,
            "completed_tasks": [], # Initialize
            "task_results": {}     # Initialize
            }
    except Exception as e: print(f"Decomposition Error: {e}"); return {"messages": [AIMessage(content=f"Decomposition Error: {e}")]}


# Node: Agent Invoker
# This node will run the pre-compiled supervisor agent
async def agent_node(state: AgentState, agent_runnable) -> Dict[str, Any]:
     """ Calls the supervisor agent runnable. """
     print("\n--- Node: Supervisor Agent ---")
     # Append task status to guide the agent
     sub = state.get("sub_tasks", []); comp = state.get("completed_tasks", [])
     pend = [t for t in sub if t not in comp]; next_task = pend[0] if pend else "None"
     all_done = sub and not pend

     if all_done:
         print("All tasks seem complete based on count. Preparing for synthesis call.")
         # Agent needs to be prompted to call synthesis tool
         task_status_prompt = f"\n\nSystem Note: All sub-tasks {sub} appear complete. Call the 'result_synthesis_tool' now with the original query and all gathered results."
     else:
         task_status_prompt = f"\n\nSystem Note: Goal='{state.get('original_query', '')}'. Tasks={sub}. Completed={comp}. Focus on pending: '{next_task}'. Review history/results, decide next action (call tool). If all done, call 'result_synthesis_tool'."

     # Create a temporary state copy to add the system note without permanently altering messages history yet
     current_messages = list(state['messages']) + [SystemMessage(content=task_status_prompt)]
     agent_input = {"messages": current_messages} # Pass only messages

     print(f"Invoking supervisor agent with {len(current_messages)} messages...")
     # The agent_runnable (created with create_agent) handles the LLM call AND tool execution
     result = await agent_runnable.ainvoke(agent_input)
     print(f"Supervisor agent output messages: {result.get('messages')}") # Debug: See what the agent returns

     # The result from create_agent should contain the new AIMessage and potential ToolMessages
     # We just need to return these messages to be added to the main state
     # LangGraph's operator.add handles appending messages correctly
     return {"messages": result["messages"]}


# Node: Process results - Simplified, mainly for tracking completion
def process_tool_results_node(state: AgentState) -> Dict[str, Any]:
    """ Associates tool results with tasks and marks tasks complete (tentatively). """
    print("\n--- Node: Process Tool Results ---")
    # --- (Logic largely same as before, ensures task_results is updated) ---
    messages = state["messages"]; last_message = messages[-1]
    if not isinstance(last_message, ToolMessage): print("Warn: No ToolMessage."); return {}
    tcid = last_message.tool_call_id; content = last_message.content; name = "?"; args = {}
    for msg in reversed(messages[:-1]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc['id'] == tcid: name = tc.get('name', '?'); args = tc.get('args', {}); break
            if name != "?": break
    print(f"Processing result for: {name}")
    tasks = state.get("sub_tasks", []); comp = state.get("completed_tasks", [])
    pend = [t for t in tasks if t not in comp]; current = pend[0] if pend else "general"
    results = state.get('task_results', {}).copy(); current_list = results.get(current, [])
    if not isinstance(current_list, list): current_list = []
    current_list.append({"tool_called": name, "tool_args": args, "tool_result": content})
    results[current] = current_list; print(f"Stored result for task: {current}")
    # Tentatively mark complete
    completed_update = []
    if current != "general" and current not in comp:
        completed_update = [current]; print(f"Marking complete (tentative): {current}")
    return {"task_results": results, "completed_tasks": completed_update}


# Node: Final Answer Extractor (optional, depends on create_agent output)
def final_answer_node(state: AgentState) -> Dict[str, Any]:
    """ Extracts the final synthesis report if available. """
    print("\n--- Node: Final Answer ---")
    messages = state["messages"]
    # Check if the last message is an AIMessage without tool calls (typical final output)
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
         print("Final answer found in last AIMessage.")
         # Keep only the final answer message
         return {"messages": [last_message]}
    else:
        # Fallback: Look for the synthesis tool result if direct answer wasn't found
        print("Final AIMessage not found, searching for synthesis tool result...")
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tcid = msg.tool_call_id
                # Check preceding messages for the AI call that invoked synthesis
                for prev_msg in reversed(messages[:messages.index(msg)]):
                    if isinstance(prev_msg, AIMessage) and prev_msg.tool_calls:
                         if any(tc['id'] == tcid and tc['name'] == 'result_synthesis_tool' for tc in prev_msg.tool_calls):
                              print("Synthesis tool result found.")
                              return {"messages": [AIMessage(content=str(msg.content))]} # Wrap result
                         break # Found the relevant AI message
            elif isinstance(msg, AIMessage) and not msg.tool_calls:
                 # If we hit an earlier AI answer without tool calls, use that? Or error?
                 print("Found an earlier AIMessage without tool calls. Using as fallback.")
                 return {"messages": [msg]}

    print("Error: Could not extract final answer.")
    return {"messages": [AIMessage(content="Error: Could not determine final answer.")]}


# --- 5. Build and Compile the Graph ---
async def compile_graph() -> StateGraph:
    """ Fetches tools, creates supervisor agent, and compiles the graph. """
    print("[Graph Builder] Compiling Coordinator Graph...")
    all_tools = await get_all_coordinator_tools()
    if not all_tools: raise ValueError("CRITICAL ERROR: No tools loaded.")

    # Define the Supervisor Prompt
    # Ensure it instructs the agent on the overall goal, task management, and when to synthesize.
    supervisor_prompt = SystemMessage(content=f"""
You are the supervisor agent coordinating a research task.
Your goal is to answer the user's original query comprehensively by breaking it down into sub-tasks and delegating them to specialist agents using the available tools.

Available Tools: {[t.name for t in all_tools]}

Workflow:
1. You have been given a list of sub-tasks generated from the user's query.
2. Address the sub-tasks sequentially. Review the state for completed tasks and results.
3. For the *first pending* sub-task, choose the most appropriate tool(s) (e.g., call_agent_2_semantic_search, call_agent_3_analysis, call_agent_4_validation) and call them with the necessary arguments. You might need multiple tool calls for one sub-task (e.g., search then analyze).
4. After a tool runs, its results will be added to the state. Review the results in the next step.
5. If you judge that a sub-task is fully addressed by the results, clearly state your reasoning for concluding the task in your response, and then proceed to decide the action for the *next* pending sub-task.
6. Once ALL sub-tasks are completed, and you have reviewed all results, your FINAL action MUST be to call the `result_synthesis_tool` with the original query and all gathered data. Do not respond directly to the user before synthesis.
7. If you encounter errors or cannot complete a task, report the issue.
""")

    # Create the Supervisor Agent Runnable
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    # Note: Pass the SystemMessage prompt directly if create_agent supports it,
    # otherwise, include it in the messages list passed to agent_node.
    # Newer versions of create_agent accept a `prompt` argument. Let's try that.
    # We might need a ChatPromptTemplate incorporating the system message and placeholders
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    prompt_template = ChatPromptTemplate.from_messages([
        supervisor_prompt,
        MessagesPlaceholder(variable_name="messages"), # The history comes here
        # Note: We are adding the task status prompt dynamically in agent_node
    ])

    try:
        # Attempt to create agent with the prompt
         supervisor_agent = create_agent(llm, all_tools, prompt=prompt_template)
         print("[Graph Builder] Supervisor agent created successfully.")
    except Exception as e:
         print(f"[Graph Builder] Error creating supervisor agent: {e}")
         # Fallback or re-raise depending on the error
         raise


    # Define the graph instance
    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("decompose", decompose_query_node)
    # Pass the created agent runnable to the agent_node
    graph_builder.add_node("supervisor_agent", lambda state: agent_node(state, supervisor_agent))
    graph_builder.add_node("process_results", process_tool_results_node)
    graph_builder.add_node("final_answer", final_answer_node)

    # Define edges
    graph_builder.set_entry_point("decompose")
    graph_builder.add_edge("decompose", "supervisor_agent")

    # Conditional edge after the supervisor agent runs
    def agent_condition(state: AgentState) -> Literal["process_results", "__end__"]:
        """ Decide whether to process tool results or end. """
        print("\n--- Edge: After Supervisor Agent ---")
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            # Check if the *only* tool call is synthesis
            if len(last_message.tool_calls) == 1 and last_message.tool_calls[0]['name'] == 'result_synthesis_tool':
                 print("Decision: Synthesis tool called -> Process Results (will lead to end)")
                 return "process_results" # Go process the synthesis call
            else:
                 print("Decision: Specialist tool(s) called -> Process Results")
                 return "process_results" # Go process specialist tool calls
        else:
            # If agent returns AIMessage *without* tool calls, it implies it finished early or errored
            print("Decision: No tool calls from supervisor -> END (or error state)")
            return "__end__" # End the graph directly

    graph_builder.add_conditional_edges(
        "supervisor_agent",
        agent_condition,
        {
            "process_results": "process_results", # If tools were called, process them
            "__end__": "__end__"              # If no tools called, end graph
        }
    )

    # After processing results, decide whether to loop back or finish
    def after_processing_condition(state: AgentState) -> Literal["supervisor_agent", "final_answer"]:
        """ Check if the synthesis tool result was just processed. """
        print("\n--- Edge: After Processing Results ---")
        # Check if the result processed was from the synthesis tool
        # Need to look at the last *appended* result in task_results or check tool name
        sub_tasks = state.get("sub_tasks", [])
        completed_tasks = state.get("completed_tasks", [])
        all_tasks_marked_complete = sub_tasks and set(sub_tasks) == set(completed_tasks)

        # More robust check: Did the *last tool message processed* correspond to synthesis?
        last_message = state["messages"][-1] # This is the ToolMessage
        is_synthesis_result = False
        if isinstance(last_message, ToolMessage):
            tool_call_id = last_message.tool_call_id
            for msg in reversed(state["messages"][:-1]):
                 if isinstance(msg, AIMessage) and msg.tool_calls:
                      if any(tc['id'] == tool_call_id and tc['name'] == 'result_synthesis_tool' for tc in msg.tool_calls):
                           is_synthesis_result = True
                           break
                      break # Found the relevant AIMessage

        if is_synthesis_result:
            print("Decision: Synthesis result processed -> Go to Final Answer Node")
            return "final_answer"
        else:
            print("Decision: Specialist result processed -> Loop back to Supervisor")
            return "supervisor_agent" # Loop back to supervisor agent

    graph_builder.add_conditional_edges(
        "process_results",
        after_processing_condition,
        {
            "supervisor_agent": "supervisor_agent", # Loop back for next step
            "final_answer": "final_answer"         # Go extract final answer
        }
    )

    # Final answer node leads to the end
    graph_builder.add_edge("final_answer", END)


    # Compile the graph
    coordinator_graph = graph_builder.compile()
    print("[Graph Builder] Coordinator graph compiled successfully.")

    try: print("\nGraph Structure:"); coordinator_graph.get_graph().print_ascii()
    except Exception as e: print(f"Could not print graph structure: {e}")

    return coordinator_graph

# --- Main block for testing compilation ---
if __name__ == '__main__':
    async def test_compilation():
        graph = await compile_graph()
        if not graph: print("Graph compilation failed during test.")

    print("Graph builder defined. Import 'compile_graph' in run_coordinator.py.")
    import asyncio
    asyncio.run(test_compilation())
