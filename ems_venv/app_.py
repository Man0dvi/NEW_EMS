import asyncio
import operator
from typing import TypedDict, List, Annotated, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# --- 1. LLM Class Definition ---

class LLM:
    """
    A simple wrapper class to encapsulate the LLM.
    This makes it easy to swap models or add logic later.
    """
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        """
        Initializes the LLM.
        Assumes OPENAI_API_KEY is set in the environment.
        """
        try:
            self.model = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                streaming=True
            )
        except ImportError:
            raise ImportError(
                "Please install langchain_openai: pip install langchain_openai"
            )
        except Exception as e:
            # Handle API key not set, etc.
            print(f"Error initializing ChatOpenAI: {e}")
            print("Please make sure your OPENAI_API_KEY is set in your environment.")
            self.model = None

    def get_model(self) -> BaseChatModel:
        """Returns the instantiated LangChain model."""
        if self.model is None:
            raise ValueError("LLM was not initialized correctly.")
        return self.model

# Global LLM instance to be used by all nodes
# We instantiate the class once here.
try:
    llm = LLM(model_name="gpt-4o").get_model()
except ValueError as e:
    print(e)
    # Define a fallback or exit
    llm = None 

# --- 2. Graph State Definition ---

class AgentState(TypedDict):
    """
    Defines the state of the graph.
    This dictionary is passed between all nodes.
    """
    # The user's initial question
    input: str
    
    # The history of messages
    # `operator.add` appends new messages to this list
    messages: Annotated[List[BaseMessage], operator.add]
    
    # A list of search queries to execute
    search_queries: List[str]
    
    # A list of retrieved documents
    retrieved_docs: List[str] # Using simple strings for this example
    
    # IDs for the retrieved documents
    document_ids: List[str]
    
    # A string describing the current step or plan
    current_step: str

# --- 3. Node Definitions ---

def entry_node(state: AgentState) -> Dict[str, Any]:
    """
    This is the first node called. It formats the user's input
    into the `messages` list as a HumanMessage.
    """
    print("--- [Entry Node] ---")
    return {
        "messages": [HumanMessage(content=state["input"])]
    }

# --- Coordinator Node ---

class Plan(BaseModel):
    """
    A Pydantic model for the coordinator's plan.
    The LLM will be forced to output JSON matching this structure.
    """
    search_queries: List[str] = Field(
        description="A list of 1-3 specific search queries to find relevant documents."
    )
    plan_description: str = Field(
        description="A brief description of the multi-step plan to answer the user's query."
    )

async def coordinator_node(state: AgentState) -> Dict[str, Any]:
    """
    This node represents the "Coordinator" or "Planner".
    It takes the user's query and generates a plan, including
    search queries for the retrieval agent.
    """
    print("--- [Coordinator Node] ---")
    
    if llm is None:
        return {"current_step": "Error: LLM not initialized."}

    # System prompt for the coordinator
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a master coordinator for a multi-agent AI system. "
                "Your job is to analyze the user's request and create a step-by-step plan. "
                "First, generate a list of search queries for the 'Web Scraper/Document Retrieval Agent' to gather information. "
                "Then, write a brief plan for the 'Deep Analysis Agent' to follow. "
                "Output *only* the JSON object with the 'search_queries' and 'plan_description' fields.",
            ),
            ("human", "{user_query}"),
        ]
    )
    
    # Create an LCEL (LangChain Expression Language) chain
    # This chain forces the LLM to output structured JSON matching our `Plan` model
    planner_chain = prompt | llm.with_structured_output(Plan)
    
    # Get the last human message
    user_query = state["messages"][-1].content
    
    try:
        # Run the chain
        plan_output = await planner_chain.ainvoke({"user_query": user_query})
        
        # This is what the node returns to the graph.
        # These values will be merged into the main `AgentState`.
        return {
            "current_step": plan_output.plan_description,
            "search_queries": plan_output.search_queries,
            "messages": [
                AIMessage(
                    content=f"Coordinator: I have created a plan. "
                            f"First, I will search for: {', '.join(plan_output.search_queries)}"
                )
            ],
        }
    except Exception as e:
        print(f"Error in coordinator_node: {e}")
        return {
            "current_step": f"Error: Could not generate plan. {e}",
            "messages": [AIMessage(content=f"Sorry, I encountered an error while planning: {e}")]
        }

# --- Retrieval Node ---

async def mock_retriever(query: str) -> Dict[str, str]:
    """
    A mock function for the "Web Scraper/Document Retrieval Agent".
    In a real project, this would use Tavily, a vectorstore, or a scraper.
    """
    print(f"  > Mock retrieving for: '{query}'")
    await asyncio.sleep(0.5) # Simulate network delay
    # Mock data
    doc_content = f"This is a mock document about '{query}'. It contains detailed analysis and data."
    doc_id = f"doc_{hash(query) % 10000}"
    return {"content": doc_content, "id": doc_id}

async def retrieval_node(state: AgentState) -> Dict[str, Any]:
    """
    This node represents the "Web Scraper/Document Retrieval Agent".
    It executes the search queries from the state.
    """
    print("--- [Retrieval Node] ---")
    
    queries = state.get("search_queries", [])
    if not queries:
        return {
            "current_step": "No search queries found. Skipping retrieval."
        }
    
    # Run all retrievals in parallel
    retrieval_tasks = [mock_retriever(query) for query in queries]
    retrieved_results = await asyncio.gather(*retrieval_tasks)
    
    # Process results
    retrieved_docs = [result["content"] for result in retrieved_results]
    document_ids = [result["id"] for result in retrieved_results]
    
    # Format a ToolMessage to log the results in the chat history
    tool_message_content = "\n\n".join(
        [
            f"--- Document (ID: {doc_id}) ---\n{doc_content}"
            for doc_id, doc_content in zip(document_ids, retrieved_docs)
        ]
    )
    
    return {
        "retrieved_docs": retrieved_docs,
        "document_ids": document_ids,
        "current_step": f"Retrieved {len(retrieved_docs)} documents.",
        # Add a ToolMessage to the history
        "messages": [
            ToolMessage(
                content=f"Retrieved {len(retrieved_docs)} documents based on queries: {', '.join(queries)}\n\n{tool_message_content}",
                tool_call_id="retrieval_tool" # A placeholder ID
            )
        ],
    }

# --- Analysis Node ---

async def analysis_node(state: AgentState) -> Dict[str, Any]:
    """
    This node represents the "Deep Analysis Agent".
    It synthesizes an answer using the retrieved documents and chat history.
    """
    print("--- [Analysis Node] ---")
    
    if llm is None:
        return {"current_step": "Error: LLM not initialized."}

    # System prompt for the analysis agent
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world-class deep analysis agent. "
                "Your job is to synthesize a comprehensive answer to the user's query "
                "based *only* on the provided context (retrieved documents) and the chat history. "
                "Do not make up information. Be thorough and clear. "
                "Cite the document IDs if possible (e.g., [doc_1234]).\n\n"
                "--- CONTEXT (Retrieved Documents) ---\n"
                "{context}\n"
                "--- END CONTEXT ---"
            ),
            # The `state["messages"]` list will be automatically populated here
            # We just need to make sure the key "messages" matches
            ("messages", "{messages}"),
        ]
    )
    
    # Create the LCEL chain
    analysis_chain = prompt | llm
    
    # Format the context
    context = "\n\n".join(
        [
            f"--- Document (ID: {doc_id}) ---\n{doc_content}"
            for doc_id, doc_content in zip(state["document_ids"], state["retrieved_docs"])
        ]
    )
    
    if not state["retrieved_docs"]:
        context = "No documents were retrieved."

    try:
        # We pass the *entire* message history to the chain
        response = await analysis_chain.ainvoke(
            {"context": context, "messages": state["messages"]}
        )
        
        return {
            "messages": [response], # Add the final AI answer
            "current_step": "Analysis complete. Final answer generated."
        }
    except Exception as e:
        print(f"Error in analysis_node: {e}")
        return {
            "messages": [AIMessage(content=f"Sorry, I encountered an error during analysis: {e}")]
        }

# --- Example of how to test the nodes (optional) ---

async def main_test():
    """
    This function demonstrates how to run the nodes sequentially
    to test them. This is *not* the graph, just a simple test.
    """
    
    # 1. Initialize state
    initial_state: AgentState = {
        "input": "How do containerization technologies like Docker and Podman differ, and what are the security implications of each?",
        "messages": [],
        "search_queries": [],
        "retrieved_docs": [],
        "document_ids": [],
        "current_step": "Start",
    }
    
    print(f"Initial State:\n{initial_state}\n")
    
    # 2. Run Entry Node
    entry_output = entry_node(initial_state)
    initial_state.update(entry_output)
    print(f"--- After Entry Node ---\nState['messages']: {initial_state['messages']}\n")
    
    # 3. Run Coordinator Node
    coord_output = await coordinator_node(initial_state)
    initial_state.update(coord_output)
    print(f"--- After Coordinator Node ---\nStep: {initial_state['current_step']}\nQueries: {initial_state['search_queries']}\n")
    
    # 4. Run Retrieval Node
    retrieval_output = await retrieval_node(initial_state)
    initial_state.update(retrieval_output)
    print(f"--- After Retrieval Node ---\nStep: {initial_state['current_step']}\nDocs: {len(initial_state['retrieved_docs'])} retrieved\n")
    
    # 5. Run Analysis Node
    analysis_output = await analysis_node(initial_state)
    initial_state.update(analysis_output)
    print(f"--- After Analysis Node ---\nStep: {initial_state['current_step']}\n")
    
    print("--- Final Answer ---")
    final_answer = initial_state["messages"][-1]
    if isinstance(final_answer, AIMessage):
        print(final_answer.content)

if __name__ == "__main__":
    # This block allows you to run `python nodes.py` to test the nodes
    
    # Make sure your OPENAI_API_KEY is set in your environment variables!
    # e.g., export OPENAI_API_KEY='your_key_here'
    
    if llm is None:
        print("LLM is not available. Exiting test.")
    else:
        print("Starting node test...")
        asyncio.run(main_test())
        
