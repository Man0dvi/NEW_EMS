# --- tools/code_discovery_tools.py ---

from typing import List, Dict, Any
from services.llm_service import LLMService
import json

# Define MAX_CHUNKS_FOR_PROMPT
MAX_CHUNKS_FOR_PROMPT = 50 # Limit number of chunks processed at once
MAX_CONTENT_FOR_PROMPT = 50000 # Limit total characters sent

async def chunk_code_semantically(filename: str, file_content: str, llm: LLMService) -> List[Dict[str, Any]]:
    """ Uses LLM to chunk a single file's content into logical structures (functions, classes, etc.). """
    print(f"[Code Discovery Tool] Semantically chunking: {filename}")
    if len(file_content) > MAX_CONTENT_FOR_PROMPT * 2: # Allow more for chunking
        print(f"Warning: Content for {filename} truncated for semantic chunking.")
        file_content = file_content[:MAX_CONTENT_FOR_PROMPT * 2] + "\n... [TRUNCATED]"

    prompt = (
        f"Analyze the source code from '{filename}':\n```\n{file_content}\n```\n\n"
        f"Divide the code into meaningful semantic chunks based on logical units like functions, classes, methods, distinct configuration blocks, or important standalone scripts/paragraphs. "
        f"For each chunk, provide: \n"
        f"- 'type': (e.g., 'function', 'class', 'method', 'config_block', 'script_block', 'comment_block')\n"
        f"- 'name': (Name of the function/class if applicable, or a generated name like 'ConfigBlock1')\n"
        f"- 'start_line': Approximate starting line number.\n"
        f"- 'end_line': Approximate ending line number.\n"
        f"- 'content': The actual code content of the chunk.\n"
        f"- 'summary': A concise one-sentence summary of the chunk's purpose.\n\n"
        f"Return the result as a JSON list of chunk objects."
    )
    response = await llm.aask(prompt)
    try:
        chunks = json.loads(response)
        # Add filename to each chunk
        for chunk in chunks:
            chunk['file'] = filename
        return chunks
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON for semantic chunks in {filename}, returning raw.")
        # Try to create a single chunk as fallback
        return [{"file": filename, "type": "file", "name": filename, "start_line": 1, "end_line": file_content.count('\n')+1, "content": file_content, "summary": "Could not parse chunks.", "llm_raw_response": response}]
    except Exception as e:
         print(f"Error parsing semantic chunks for {filename}: {e}")
         return [{"file": filename, "error": str(e), "llm_raw_response": response}]


async def enrich_chunks_with_metadata(chunks: List[Dict[str, Any]], llm: LLMService) -> List[Dict[str, Any]]:
    """ Uses LLM to add role, dependencies, and business context metadata to chunks. """
    print(f"[Code Discovery Tool] Enriching {len(chunks)} chunks with metadata...")

    # Process chunks in batches to avoid overly long prompts
    enriched_chunks_all = []
    chunks_batch = chunks[:MAX_CHUNKS_FOR_PROMPT] # Process first batch
    if len(chunks) > MAX_CHUNKS_FOR_PROMPT:
         print(f"Warning: Processing only the first {MAX_CHUNKS_FOR_PROMPT} chunks for enrichment.")

    # Prepare simplified chunk representation for prompt
    prompt_chunks = []
    for i, chunk in enumerate(chunks_batch):
        prompt_chunks.append({
            "id": i, # Add an ID for mapping results back
            "file": chunk.get('file', 'unknown'),
            "type": chunk.get('type', 'unknown'),
            "name": chunk.get('name', 'unknown'),
            "summary": chunk.get('summary', ''),
            "content_preview": chunk.get('content', '')[:300] + "..." # Send only a preview
        })

    prompt = (
        f"Analyze the following code chunks:\n{json.dumps(prompt_chunks, indent=2)}\n\n"
        f"For each chunk (identified by 'id'), determine its primary 'code_role' (e.g., 'API Endpoint', 'Business Logic', 'Data Model', 'Utility Function', 'Configuration', 'Testing', 'Documentation'). "
        f"Also, list its main 'dependencies' (key libraries or other internal modules it seems to rely on based on its content/summary). "
        f"Finally, provide a very brief 'business_context' (1 sentence explaining its likely relevance or risk from a business/product perspective, e.g., 'Handles user authentication', 'Core calculation engine', 'Test utility with no direct impact').\n\n"
        f"Return a JSON list where each object contains 'id' and the new keys: 'code_role', 'dependencies' (list of strings), 'business_context'."
    )
    response = await llm.aask(prompt)
    try:
        enrichment_data = json.loads(response)
        # Merge enrichment data back into original chunks based on id
        enrichment_map = {item['id']: item for item in enrichment_data if 'id' in item}

        for i, chunk in enumerate(chunks_batch):
            if i in enrichment_map:
                chunk['code_role'] = enrichment_map[i].get('code_role', 'Unknown')
                chunk['dependencies'] = enrichment_map[i].get('dependencies', [])
                chunk['business_context'] = enrichment_map[i].get('business_context', 'N/A')
            else: # Handle missing enrichment
                chunk['code_role'] = 'Error'
                chunk['dependencies'] = []
                chunk['business_context'] = 'Failed to enrich.'
        enriched_chunks_all.extend(chunks_batch)

        # Handle remaining chunks if any (could add loop here for full processing)
        if len(chunks) > MAX_CHUNKS_FOR_PROMPT:
             print("Skipping enrichment for remaining chunks.")
             enriched_chunks_all.extend(chunks[MAX_CHUNKS_FOR_PROMPT:]) # Add unprocessed chunks

        return enriched_chunks_all

    except json.JSONDecodeError:
        print("Warning: Could not parse JSON for enrichment, returning original chunks.")
        return chunks # Return original chunks if parsing fails
    except Exception as e:
         print(f"Error during chunk enrichment: {e}")
         # Add error markers to chunks
         for chunk in chunks_batch:
              chunk['code_role'] = 'Error'; chunk['dependencies'] = []; chunk['business_context'] = f'Enrichment failed: {e}'
         return chunks


async def generate_persona_insights(chunks: List[Dict[str, Any]], llm: LLMService, persona: str) -> List[Dict[str, Any]]:
    """ Uses LLM to generate persona-specific insights (SDE, PM) for enriched chunks. """
    print(f"[Code Discovery Tool] Generating insights for persona: {persona} on {len(chunks)} chunks...")

    # Process in batches
    persona_chunks_all = []
    chunks_batch = chunks[:MAX_CHUNKS_FOR_PROMPT]
    if len(chunks) > MAX_CHUNKS_FOR_PROMPT:
         print(f"Warning: Processing only first {MAX_CHUNKS_FOR_PROMPT} chunks for persona insights.")

    # Prepare simplified chunk representation
    prompt_chunks = []
    for i, chunk in enumerate(chunks_batch):
        prompt_chunks.append({
            "id": i,
            "file": chunk.get('file', '?'),
            "type": chunk.get('type', '?'),
            "name": chunk.get('name', '?'),
            "summary": chunk.get('summary', '?'),
            "code_role": chunk.get('code_role', '?'),
            "dependencies": chunk.get('dependencies', []),
            "business_context": chunk.get('business_context', '?'),
        })

    # Persona-specific instructions
    if persona.lower() == 'sde':
        persona_instructions = "Focus on technical details: complexity, dependencies, potential refactoring needs, testing implications, and integration points relevant to a Software Development Engineer."
    elif persona.lower() == 'pm':
        persona_instructions = "Focus on product/business implications: user features, business logic, potential risks/impact on roadmap, areas needing clarification, and alignment with product goals relevant to a Product Manager."
    else:
        persona_instructions = "Provide a general overview of the chunk's relevance."

    prompt = (
        f"Analyze the following enriched code chunks from the perspective of a '{persona}'.\n"
        f"{json.dumps(prompt_chunks, indent=2)}\n\n"
        f"For each chunk (identified by 'id'), provide a concise 'persona_insight' (1-2 sentences) explaining its relevance and key takeaways for this persona. {persona_instructions}\n\n"
        f"Return a JSON list where each object contains only 'id' and 'persona_insight'."
    )
    response = await llm.aask(prompt)
    try:
        insight_data = json.loads(response)
        insight_map = {item['id']: item.get('persona_insight', 'N/A') for item in insight_data if 'id' in item}

        # Add insights back to the original chunks
        for i, chunk in enumerate(chunks_batch):
            chunk[f'insight_{persona.lower()}'] = insight_map.get(i, 'Failed to generate insight.')
        persona_chunks_all.extend(chunks_batch)

        if len(chunks) > MAX_CHUNKS_FOR_PROMPT:
             print("Skipping persona insights for remaining chunks.")
             persona_chunks_all.extend(chunks[MAX_CHUNKS_FOR_PROMPT:])

        return persona_chunks_all

    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON for persona insights ({persona}), adding raw.")
        # Add raw response as insight to first chunk as fallback
        if chunks_batch: chunks_batch[0][f'insight_{persona.lower()}'] = f"Raw LLM Response: {response}"
        return chunks # Return original chunks
    except Exception as e:
         print(f"Error during persona insight generation ({persona}): {e}")
         for chunk in chunks_batch: chunk[f'insight_{persona.lower()}'] = f'Insight generation failed: {e}'
         return chunks
