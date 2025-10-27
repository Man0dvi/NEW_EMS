# --- all_tools_server.py ---
# Consolidates tools for Agents 1, 2, 3, and 4 into a single MCP server.

import os
import re
import itertools
import requests
import whois
import datetime
import yake
import json # Ensure json is imported if needed for tool outputs/parsing
from typing import List, Dict, Any, Literal
from dotenv import load_dotenv

# MCP and Server
from mcp.server.fastmcp import FastMCP
import uvicorn # Using uvicorn directly for clarity on port

# LangChain components (consolidated imports)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_experimental.text_splitter import SemanticChunker # Using Semantic
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

# 1. --- INITIAL SETUP & CONFIG ---
load_dotenv()

# Database Config (Used by Agents 2, 3, 4 internals)
PG_CONNECTION_STRING = "postgresql+psycopg2://vectoruser:vectorpass@localhost:5433/vectordb"
COLLECTION_NAME = "deep_research_docs"

# 2. --- INITIALIZE SHARED RESOURCES (LLMs, DB, etc.) ---
try:
    # LLMs
    coordinator_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY")) # For Agent 1 Tools
    analysis_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY")) # For Agent 3 Tools
    heavy_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY")) # For Agent 4 Tools (complex)
    fast_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY")) # For Agent 4 Tools (simple)

    # Embedding model
    embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    # Keyword Extractor
    kw_extractor = yake.KeywordExtractor(n=1, top=20, dedupLim=0.9)

    # Database Engine & Vector Store
    db_engine = create_engine(PG_CONNECTION_STRING)
    vector_store = PGVector(
        engine=db_engine,
        embedding_function=embeddings_model,
        collection_name=COLLECTION_NAME,
        use_jsonb=True,
    )
    print("[All Tools Server] Shared resources initialized.")

except Exception as e:
    print(f"[All Tools Server] ERROR during initialization: {e}")
    exit(1)

# --- 3. CREATE NAMESPACED MCP INSTANCES ---
# One instance for each agent's toolset
coordinator_mcp = FastMCP("Coordinator_Tools")
rag_mcp = FastMCP("RAG_and_Web_Agent")
analysis_mcp = FastMCP("Deep_Analysis_Agent")
validation_mcp = FastMCP("Fact_Checking_Agent")

# --- 4. DEFINE ALL TOOLS (Decorated with correct MCP instance) ---

# --- Agent 1 Tools ---
@coordinator_mcp.tool()
def query_decomposition_tool(user_query: str) -> List[str]:
    """
    Breaks down a complex user research query into smaller, manageable sub-tasks.
    Each sub-task should be a specific question or research step.
    Args: user_query (str): The original complex query from the user.
    """
    print(f"[Coordinator Tool] Decomposing query: '{user_query[:100]}...'")
    prompt = f"""
    You are an expert research assistant... decompose... into a list of specific, answerable sub-queries... Aim for 3-6... Output *only* a Python list of strings.

    Example Input: "How did COVID-19 impact renewable energy investment..."
    Example Output: ["Identify economic impacts...", "Analyze investment trends...", ...]

    User Query: "{user_query}"
    Decomposed Sub-queries (Python list of strings):
    """
    try:
        response = coordinator_llm.invoke([SystemMessage(content=prompt)])
        raw_list_str = response.content.strip()
        if raw_list_str.startswith('[') and raw_list_str.endswith(']'):
             import ast
             sub_tasks = ast.literal_eval(raw_list_str)
             if isinstance(sub_tasks, list) and all(isinstance(item, str) for item in sub_tasks):
                  print(f"[Coordinator Tool] Decomposition successful: {len(sub_tasks)} sub-tasks.")
                  return sub_tasks
             else: raise ValueError("LLM did not return a valid list of strings.")
        else:
             print("[Coordinator Tool] Warning: LLM did not return list format.")
             return [raw_list_str]
    except Exception as e:
        print(f"[Coordinator Tool] Error during query decomposition: {e}")
        return [f"Address original query: {user_query}"]

@coordinator_mcp.tool()
def result_synthesis_tool(original_query: str, all_gathered_data: Dict[str, Any]) -> str:
    """
    Synthesizes a final, comprehensive answer based on all gathered data.
    Args: original_query (str): The initial user query.
          all_gathered_data (Dict[str, Any]): Dictionary of findings keyed by sub-task.
    """
    print(f"[Coordinator Tool] Synthesizing results for: '{original_query[:100]}...'")
    data_summary = "\n\n## Research Findings:\n"
    # Format data nicely for the prompt
    for task, results_list in all_gathered_data.items():
        data_summary += f"\n### Sub-Task: {task}\n"
        if isinstance(results_list, list):
            for result_item in results_list:
                tool_name = result_item.get("tool_called", "unknown")
                tool_result = result_item.get("tool_result", "N/A")
                res_str = str(tool_result)
                if len(res_str) > 300: res_str = res_str[:300] + "..."
                data_summary += f"- **{tool_name}**: {res_str}\n"
        else: # Handle non-list results just in case
            res_str = str(results_list)
            if len(res_str) > 300: res_str = res_str[:300] + "..."
            data_summary += f"- Findings: {res_str}\n"

    prompt = f"""
    You are a research report writer... synthesize... answer to the user's original query.
    Original User Query: "{original_query}"
    Instructions: Address query directly. Integrate findings logically. Cite sources mentioned. Incorporate confidence scores. Structure clearly. Be concise but thorough. Use *only* provided findings.
    Provided Research Findings: {data_summary}
    ---
    Final Synthesized Report:
    """
    try:
        response = coordinator_llm.invoke([SystemMessage(content=prompt)])
        print("[Coordinator Tool] Synthesis complete.")
        return response.content
    except Exception as e:
        print(f"[Coordinator Tool] Error during result synthesis: {e}")
        return f"Error synthesizing report: {e}\n\nRaw Findings:\n{data_summary}"

# --- Agent 2 Tools ---
@rag_mcp.tool()
def web_scrape_and_ingest(urls: List[str] | None = None) -> str:
    """
    Tool 1 (Agent 2): (SYNC) Scrapes content, SEMANTICALLY CHUNKS, extracts keywords,
    and ingests everything into the vector database. Use ONLY when asked to update/add sources.
    """
    global vector_store, embeddings_model # Ensure globals are accessible
    print("[RAG Tool] WebScrape (Semantic Chunking)")
    if urls: urls_to_scrape = urls; print(f"Scraping {len(urls)} URLs...")
    else: urls_to_scrape = PREDEFINED_URL_LIST; print(f"Scraping {len(urls_to_scrape)} predefined URLs...")

    loader = WebBaseLoader(web_paths=urls_to_scrape)
    try: docs_list = loader.load()
    except Exception as e: return f"Error loading URLs: {e}"
    if not docs_list: return "Error: No documents loaded."

    text_splitter = SemanticChunker(embeddings_model, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95)
    chunks = []
    for doc in docs_list:
         semantic_chunks_texts = text_splitter.split_text(doc.page_content)
         print(f"  - Split doc '{doc.metadata.get('source', '')[:50]}...' into {len(semantic_chunks_texts)} chunks.")
         for i, chunk_text in enumerate(semantic_chunks_texts):
             chunk_metadata = doc.metadata.copy(); chunk_metadata["semantic_chunk_index"] = i
             # Import Document class if not already imported
             from langchain_core.documents import Document
             chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
    print(f"Total semantic chunks: {len(chunks)}.")
    if not chunks: return "No text after chunking."

    for chunk in chunks:
        if len(chunk.page_content.split()) > 5:
            keywords_tuples = kw_extractor.extract_keywords(chunk.page_content)
            chunk.metadata["keywords"] = [kw[0].lower() for kw in keywords_tuples]
        else: chunk.metadata["keywords"] = []

    try:
        print(f"Adding {len(chunks)} chunks to vector store...")
        vector_store.add_documents(chunks)
    except Exception as e: return f"Error: DB ingestion failed. {e}"
    return f"Successfully scraped, chunked, ingested {len(urls_to_scrape)} URLs. Added {len(chunks)} chunks."

@rag_mcp.tool()
def semantic_search_tool(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """Tool 2 (Agent 2): (SYNC) Performs semantic search. Finds conceptually similar docs."""
    global vector_store
    print(f"[RAG Tool] Semantic search for: '{query}'")
    try:
        results = vector_store.similarity_search_with_score(query, k=k)
        formatted = [{"text_content": doc.page_content, "source": doc.metadata.get('source', 'N/A'), "similarity_score": score} for doc, score in results]
        if not formatted: return [{"status": "No relevant documents found."}]
        return formatted
    except Exception as e: return [{"error": f"Error during semantic search: {e}"}]

@rag_mcp.tool()
def keyword_search_tool(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Tool 3 (Agent 2): (SYNC) Performs keyword search on metadata. Finds exact term matches."""
    global db_engine, vector_store
    print(f"[RAG Tool] Keyword search for: '{query}'")
    try:
        table_name = vector_store.embedding_table_name
        query_keywords = list(set(query.lower().split()))
        sql = text(f"SELECT document, cmetadata FROM {table_name} WHERE cmetadata -> 'keywords' ?| array[:query_keywords] LIMIT :limit")
        formatted = []
        with db_engine.connect() as conn:
            res = conn.execute(sql, {"query_keywords": query_keywords, "limit": limit})
            for row in res.mappings():
                formatted.append({"text_content": row['document'], "source": row['cmetadata'].get('source', 'N/A'), "matching_keywords": [kw for kw in row['cmetadata'].get('keywords', []) if kw in query_keywords]})
        if not formatted: return [{"status": "No relevant documents found."}]
        return formatted
    except Exception as e: return [{"error": f"Error during keyword search: {e}"}]


# --- Agent 3 Tools ---
# Internal helpers needed by Agent 3's tools
def _internal_semantic_search_a3(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Internal RAG for Agent 3 tools"""
    global vector_store
    print(f"[Agent 3 Internal] Semantic search for '{query}' (k={k})")
    try:
        results = vector_store.similarity_search_with_score(query, k=k)
        return [{"source": doc.metadata.get('source', 'N/A'), "text": doc.page_content} for doc, score in results]
    except Exception as e: return [{"source": "Error", "text": f"Error retrieving context: {e}"}]

@analysis_mcp.tool()
def comparative_analysis_tool(topic1: str, topic2: str) -> str:
    """Tool 1 (Agent 3): Retrieves info and compares/contrasts topic1 and topic2."""
    print(f"[Analysis Tool] Comparative Analysis: '{topic1}' vs '{topic2}'")
    context1_list = _internal_semantic_search_a3(topic1)
    context2_list = _internal_semantic_search_a3(topic2)
    context1 = "\n---\n".join([c['text'] for c in context1_list if 'Error' not in c['source']])
    context2 = "\n---\n".join([c['text'] for c in context2_list if 'Error' not in c['source']])
    combined_context = f"Context on '{topic1}':\n{context1}\n\nContext on '{topic2}':\n{context2}"
    prompt = f"""You are a research analyst comparing '{topic1}' and '{topic2}' based ONLY on the provided context. Focus on differences if contradictory, else nuanced comparisons. Structure clearly. No outside knowledge. Context:\n---\n{combined_context}\n---\nAnalysis:"""
    try: return analysis_llm.invoke([SystemMessage(content=prompt)]).content
    except Exception as e: return f"Error during comparative analysis: {e}"

@analysis_mcp.tool()
def trend_analysis_tool(topic: str) -> str:
    """Tool 2 (Agent 3): Retrieves info and analyzes temporal trends for the topic."""
    print(f"[Analysis Tool] Trend Analysis: '{topic}'")
    context_list = _internal_semantic_search_a3(topic, k=10)
    context = "\n---\n".join([c['text'] for c in context_list if 'Error' not in c['source']])
    prompt = f"""You are a data analyst identifying temporal trends for '{topic}' based ONLY on the context. Look for changes over time, dates, sequences. If none, state that clearly. No outside knowledge. Context:\n---\n{context}\n---\nTrend Analysis:"""
    try: return analysis_llm.invoke([SystemMessage(content=prompt)]).content
    except Exception as e: return f"Error during trend analysis: {e}"

@analysis_mcp.tool()
def causal_reasoning_tool(effect_or_event: str, potential_causes_query: str) -> str:
    """Tool 3 (Agent 3): Retrieves info and identifies cause-effect relationships."""
    print(f"[Analysis Tool] Causal Reasoning: '{effect_or_event}'")
    context_list = _internal_semantic_search_a3(f"{effect_or_event} and {potential_causes_query}", k=5)
    context = "\n---\n".join([c['text'] for c in context_list if 'Error' not in c['source']])
    prompt = f"""You are a logic expert identifying cause-effect links for '{effect_or_event}' based ONLY on context. State causes/effects. If none mentioned, state that. No outside knowledge. Context:\n---\n{context}\n---\nCausal Analysis:"""
    try: return analysis_llm.invoke([SystemMessage(content=prompt)]).content
    except Exception as e: return f"Error during causal reasoning: {e}"

@analysis_mcp.tool()
def statistical_analysis_tool(quantitative_query: str) -> str:
    """Tool 4 (Agent 3): Retrieves info and extracts/calculates quantitative data."""
    print(f"[Analysis Tool] Statistical Analysis: '{quantitative_query}'")
    context_list = _internal_semantic_search_a3(quantitative_query, k=5)
    context = "\n---\n".join([c['text'] for c in context_list if 'Error' not in c['source']])
    prompt = f"""You are a data processor answering: "{quantitative_query}". Extract numbers/facts ONLY from context. Calculate sums/averages if needed based ONLY on data. Format clearly. If data unavailable, state that. No outside knowledge. Context:\n---\n{context}\n---\nStatistical Answer:"""
    try: return analysis_llm.invoke([SystemMessage(content=prompt)]).content
    except Exception as e: return f"Error during statistical analysis: {e}"


# --- Agent 4 Tools ---
# Helper needed by Agent 4
def _normalize_domain_a4(url: str) -> str:
    """Helper for Agent 4"""
    try:
        match = re.search(r"^(?:https?:\/\/)?(?:[^@\n]+@)?(?:www\.)?([^:\/\n?]+)", url)
        if match: return match.group(1)
        else: parts = url.split('/'); return parts[0] if len(parts) > 0 and '.' in parts[0] else url
    except Exception: return url

@validation_mcp.tool()
def source_credibility_checker(source_url: str) -> Dict[str, Any]:
    """Tool 1 (Agent 4): Evaluates source reliability (OS version)."""
    global TRUSTED_SOURCE_LIST # Assuming TRUSTED_SOURCE_LIST is defined above or imported
    print(f"[Validation Tool] Credibility Check: '{source_url}'")
    # --- (Copy the full implementation from the previous agent_4_tools_os.py) ---
    domain = _normalize_domain_a4(source_url)
    justification_parts = []
    score = 5 # Start neutral
    TRUSTED_SOURCE_LIST = { # Define again locally or ensure accessible
         "arxiv.org": {"score": 9, "justification": "High-trust academic preprint server."},
         "acm.org": {"score": 9, "justification": "Respected computer science publisher."},
         "nature.com": {"score": 10, "justification": "Top-tier scientific journal."},
         "lilianweng.github.io": {"score": 8, "justification": "Well-regarded AI researcher blog."},
         "wikipedia.org": {"score": 6, "justification": "Good general overview, but not a primary source."}
    }

    if domain in TRUSTED_SOURCE_LIST: return TRUSTED_SOURCE_LIST[domain]

    url_to_check = source_url if source_url.startswith('http') else 'https://' + source_url
    try:
        response = requests.get(url_to_check, timeout=5, allow_redirects=True, verify=True)
        if response.url.startswith("https://"): score += 1; justification_parts.append("Uses HTTPS.")
        else: score -= 2; justification_parts.append("Does not use HTTPS.")
    except requests.exceptions.SSLError: score -= 3; justification_parts.append("HTTPS certificate validation failed.")
    except requests.exceptions.RequestException as e: score -= 1; justification_parts.append(f"Could not verify HTTPS ({type(e).__name__}).")

    try:
        whois_domain = _normalize_domain_a4(url_to_check)
        domain_info = whois.whois(whois_domain)
        creation_date = domain_info.creation_date
        if isinstance(creation_date, list): creation_date = creation_date[0]
        if isinstance(creation_date, datetime.datetime):
            now = datetime.datetime.now(creation_date.tzinfo)
            age_years = (now - creation_date).days / 365.25
            justification_parts.append(f"Domain registered for {age_years:.1f} years.")
            if age_years > 5: score += 2
            elif age_years > 1: score += 1
            else: score -= 1
        else: justification_parts.append("Could not parse domain creation date."); score -=1
    except whois.parser.PywhoisError: justification_parts.append(f"Could not retrieve WHOIS data for {whois_domain}."); score -= 1
    except Exception as e: justification_parts.append(f"Error during WHOIS lookup for {whois_domain}: {type(e).__name__}"); score -= 1

    final_score = max(1, min(score, 10))
    final_justification = " ".join(justification_parts) if justification_parts else "Basic checks inconclusive."
    return {"score": final_score, "justification": final_justification}
    # --- (End of copied implementation) ---

@validation_mcp.tool()
def cross_reference_validator(claim: str, query_for_context: str) -> Dict[str, Any]:
    """Tool 2 (Agent 4): Retrieves context and verifies claim internally using NLI."""
    print(f"[Validation Tool] Cross-Reference: '{claim}' using context for '{query_for_context}'")
    contexts = _internal_semantic_search_a3(query_for_context, k=5) # Use Agent 3's helper
    if not contexts or "Error" in contexts[0]["source"]: return {"claim": claim, "error": "Could not retrieve context."}
    # --- (Copy the NLI loop from the previous agent_4_tools_os.py) ---
    internal_results = {"supporting": [], "refuting": [], "neutral": []}
    for i, context_item in enumerate(contexts):
        context_text = context_item.get('text', ''); source = context_item.get('source', f'Ctx{i+1}')
        prompt = f'Analyze relationship: Claim: "{claim}" Context: "{context_text}" Based *only* on context, does it: SUPPORT, REFUTE, or NEUTRAL? Single word answer.'
        try:
            if not context_text or not context_text.strip(): internal_results["neutral"].append(source + " (Empty)"); continue
            response = heavy_llm.invoke([SystemMessage(content=prompt)]).content.upper()
            if "SUPPORT" in response: internal_results["supporting"].append(source)
            elif "REFUTE" in response: internal_results["refuting"].append(source)
            else: internal_results["neutral"].append(source)
        except Exception as e: internal_results["neutral"].append(source + f" (NLI Error: {e})")
    # --- (End of copied loop) ---
    return {"claim": claim, "internal_validation": internal_results}

@validation_mcp.tool()
def contradiction_detector(query_for_context: str, num_docs_to_check: int = 5) -> List[Dict[str, Any]]:
    """Tool 3 (Agent 4): Retrieves context and finds internal contradictions."""
    k = max(2, min(num_docs_to_check, 10))
    print(f"[Validation Tool] Contradiction Detect: context for '{query_for_context}' (top {k})")
    contexts = _internal_semantic_search_a3(query_for_context, k=k) # Use Agent 3's helper
    if not contexts or "Error" in contexts[0]["source"] or len(contexts) < 2: return []
    # --- (Copy the contradiction loop from the previous agent_4_tools_os.py) ---
    contradictions = []
    for (item1, item2) in itertools.combinations(contexts, 2):
        text1 = item1.get('text', ''); text2 = item2.get('text', '')
        source1 = item1.get('source', 'Src1'); source2 = item2.get('source', 'Src2')
        if not text1 or not text2 or not text1.strip() or not text2.strip(): continue
        prompt = f'Analyze contradiction: A (from {source1}): "{text1}" B (from {source2}): "{text2}" Do they directly contradict? Answer "YES" or "NO", with explanation.'
        try:
            response = heavy_llm.invoke([SystemMessage(content=prompt)]).content
            if response.strip().upper().startswith("YES"):
                print(f"CONTRADICTION FOUND: {source1} vs {source2}")
                contradictions.append({"sources": [source1, source2], "explanation": response})
        except Exception as e: print(f"Error checking contradiction {source1} vs {source2}: {e}")
    # --- (End of copied loop) ---
    return contradictions

@validation_mcp.tool()
def confidence_scorer(claim: str, cross_ref_results: Dict[str, Any], contradiction_results: List[Dict[str, Any]], credibility_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tool 4 (Agent 4): Assigns final confidence score based on evidence."""
    print(f"[Validation Tool] Confidence Score for: '{claim}'")
    # --- (Copy the implementation from the previous agent_4_tools_os.py) ---
    internal_val = cross_ref_results.get('internal_validation', {}); supporting_src = internal_val.get('supporting', [])
    refuting_src = internal_val.get('refuting', []); neutral_src = internal_val.get('neutral', [])
    evidence_summary = f"""Claim: "{claim}" Evidence Report:
    1. Internal Cross-Ref: Supporting={supporting_src}, Refuting={refuting_src}, Neutral={neutral_src}
    2. Internal Contradictions: Found {len(contradiction_results)}. {'Details: ' + str(contradiction_results) if contradiction_results else ''}
    3. Source Credibility: {credibility_reports}"""
    prompt = f"""Assign confidence score (0-100) based ONLY on report. Guidelines: High internal support (credible src>=7, no contradictions) = 80+. Internal refutation (credible src>=7) = <20. Contradiction between credible sources = max 50. Support only low credibility (src<=4) = max 40. Mixed/medium credibility (src 5-6) = 40-60. Lack of evidence = 10-30. Report:\n{evidence_summary}\nFormat:\nScore: [score]\nJustification: [justification...]"""
    try:
        response = fast_llm.invoke([SystemMessage(content=prompt)]).content
        score_match = re.search(r"Score: (\d+)", response); just_match = re.search(r"Justification: ([\s\S]*)", response)
        score = int(score_match.group(1)) if score_match else 0; justification = just_match.group(1).strip() if just_match else "N/A."
        score = max(0, min(score, 100))
        return {"score": score, "justification": justification}
    except Exception as e: return {"score": 0, "justification": f"Error scoring: {e}"}
    # --- (End of copied implementation) ---


# --- 5. MERGE & RUN THE SERVER ---
# Combine all namespaced apps into one main app for Uvicorn
from fastapi import FastAPI
app = FastAPI(title="Multi-Agent Tool Server")

# Mount each agent's MCP app under a namespace path
# NOTE: The client needs to call tools using namespace prefixes,
# e.g., "Coordinator_Tools.query_decomposition_tool"
app.mount("/coordinator", coordinator_mcp.build_app(transport="streamable-http"))
app.mount("/rag", rag_mcp.build_app(transport="streamable-http"))
app.mount("/analysis", analysis_mcp.build_app(transport="streamable-http"))
app.mount("/validation", validation_mcp.build_app(transport="streamable-http"))

if __name__ == "__main__":
    print("[All Tools Server] Starting consolidated server via Uvicorn on port 8000...")
    # Run the combined FastAPI app using Uvicorn on port 8000
    uvicorn.run(app, host="localhost", port=8000)
