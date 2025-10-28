# --- services/llm_service.py ---

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class LLMService:
    """
    A simple wrapper for LangChain LLM interactions.
    Handles API key loading and provides sync/async ask methods.
    """
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass it during initialization.")

        try:
            # Initialize the ChatOpenAI model
            self.llm = ChatOpenAI(
                model=model,
                temperature=0, # Low temperature for more deterministic analysis
                api_key=self.api_key
            )
            print(f"[LLMService] Initialized with model: {model}")
        except Exception as e:
            print(f"[LLMService] Error initializing ChatOpenAI: {e}")
            raise

    def ask(self, prompt: str, system_prompt: str | None = None) -> str:
        """ Synchronous call to the LLM. """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"[LLMService] Error during sync invoke: {e}")
            return f"Error: LLM call failed - {e}"

    async def aask(self, prompt: str, system_prompt: str | None = None) -> str:
        """ Asynchronous call to the LLM. """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            print(f"[LLMService] Error during async invoke: {e}")
            return f"Error: LLM call failed - {e}"

# Example Usage (optional, for testing)
if __name__ == '__main__':
    # Make sure you have a .env file with OPENAI_API_KEY
    try:
        llm_service = LLMService()
        sync_response = llm_service.ask("What is the capital of France?")
        print(f"Sync Response: {sync_response}")

        import asyncio
        async def run_async_test():
            async_response = await llm_service.aask("Briefly explain LangGraph.")
            print(f"Async Response: {async_response}")
        asyncio.run(run_async_test())
    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"An error occurred: {e}")
