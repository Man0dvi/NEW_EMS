# --- run_coordinator.py (Takes User Input - No changes needed) ---

import asyncio
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import json

# Import the graph compiler function
from graph_builder import compile_graph # Import from the refactored file

load_dotenv()

# Function to pretty print messages/tool outputs (same as before)
def format_output(data):
    try:
        if isinstance(data, (str)) and data.strip().startswith(('[','{')): data=json.loads(data)
        output_str = json.dumps(data, indent=2) if isinstance(data,(dict,list)) else str(data)
    except: output_str = str(data)
    return (output_str[:500] + '...') if len(output_str) > 500 else output_str


async def main():
    print("Compiling the Coordinator Graph...")
    coordinator_graph = await compile_graph()

    if not coordinator_graph:
        print("Exiting due to graph compilation error.")
        return

    print("\n" + "="*50)
    print(" Deep Research Assistant Coordinator Initialized (create_agent version) ")
    print(" Enter your complex research query below.")
    print(" Type 'quit', 'exit', or 'bye' to end.")
    print("="*50)

    while True:
        # --- Get User Input ---
        input_query = input("\n👤 User Query: ")
        if input_query.lower() in ["quit", "exit", "bye"]:
            print("👋 Assistant: Goodbye!")
            break
        if not input_query.strip(): continue

        print(f"\n--- Running Coordinator for Query: --- \n'{input_query}'\n" + "="*40)

        initial_state = {
            "messages": [HumanMessage(content=input_query)],
            # Reset state for each new query
            "original_query": None, "sub_tasks": None,
            "completed_tasks": [], "task_results": {}
        }
        config = {"recursion_limit": 25}

        try:
            print("\n--- Streaming Events ---")
            final_answer = None
            async for event in coordinator_graph.astream_events(initial_state, config=config, version="v1"):
                kind = event["event"]; name = event.get("name", "")
                # --- (Streaming print logic remains the same) ---
                if kind == "on_chain_start": print(f"\n>> Start: {name}")
                elif kind == "on_chain_end":
                    print(f"<< End: {name}")
                    if name == "__root__": # Check for overall graph end
                         output = event["data"].get("output")
                         # The final state should now be in output['messages'][-1]
                         if output and output.get("messages"): final_answer = output["messages"][-1].content
                elif kind == "on_tool_start": print(f"   >>> Tool Start: {event['data'].get('name')}")
                elif kind == "on_tool_end":
                    print(f"   <<< Tool End ({event['data'].get('name')}):")
                    output = event["data"].get("output"); print(f"       {format_output(output)}")
                sys.stdout.flush()

            # --- Print Final Answer ---
            print("\n" + "="*40 + "\n✅ FINAL ANSWER:\n" + "="*40)
            if final_answer: print(final_answer)
            else: # Fallback
                 print("Final answer not captured cleanly. Fetching final state...")
                 final_state = await coordinator_graph.ainvoke(initial_state, config=config)
                 if final_state and final_state.get('messages'): print(final_state['messages'][-1].content)
                 else: print("[Error retrieving final answer]")

        except Exception as e:
            print(f"\n\n!!! Graph Execution Error: {e} !!!")
            import traceback; traceback.print_exc()
        finally:
             print("\n" + "="*50) # Separator

# --- (Main execution block is the same) ---
if __name__ == "__main__":
    print("Starting Coordinator Run (create_agent Supervisor)...")
    print("Ensure the consolidated server (all_tools_server.py) is running (likely on port 8000).")
    try: asyncio.run(main())
    except KeyboardInterrupt: print("\nExecution interrupted.")
