# --- all_tools_server.py (Consolidated, Single mcp.run) ---

import os
import re
import itertools
import requests
import whois
import datetime
import yake
import json
from typing import List, Dict, Any, Literal
from dotenv import load_dotenv

# MCP and Server
from mcp.server.fastmcp import FastMCP
# REMOVED: import uvicorn
# REMOVED: from fastapi import FastAPI

# LangChain components (consolidated imports)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text
from langchain_core.documents import Document # Ensure Document is imported

# 1. --- INITIAL SETUP & CONFIG ---
load_dotenv()

# Database Config
PG_CONNECTION_STRING = "postgresql+psycopg2://vectoruser:vectorpass@localhost:5433/vectordb"
COLLECTION_NAME = "deep_research_docs"

# Predefined URL List (Agent 2)
PREDEFINED_URL_LIST = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2024-02-05-human-data-distillation/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-rag/"
]

# Trusted Source List (Agent 4)
TRUSTED_SOURCE_LIST = {
    "arxiv.org": {"score": 9, "justification": "High-trust academic preprint server."},
    "acm.org": {"score": 9, "justification": "Respected computer science publisher."},
    "nature.com": {"score": 10, "justification": "Top-tier scientific journal."},
    "lilianweng.github.io": {"score": 8, "justification": "Well-regarded AI researcher blog."},
    "wikipedia.org": {"score": 6, "justification": "Good general overview, but not a primary source."}
}

# --- 2. --- INITIALIZE SHARED RESOURCES ---
try:
    # LLMs
    coordinator_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    analysis_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    heavy_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    fast_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    kw_extractor = yake.KeywordExtractor(n=1, top=20, dedupLim=0.9)
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

# --- 3. --- CREATE NAMESPACED MCP INSTANCES ---
coordinator_mcp = FastMCP("Coordinator_Tools")
rag_mcp = FastMCP("RAG_and_Web_Agent")
analysis_mcp = FastMCP("Deep_Analysis_Agent")
validation_mcp = FastMCP("Fact_Checking_Agent")

# --- 4. --- DEFINE ALL TOOLS (Decorated with correct MCP instance) ---

# --- Agent 1 Tools (Use @coordinator_mcp.tool()) ---
@coordinator_mcp.tool()
def query_decomposition_tool(user_query: str) -> List[str]:
    # ... (Implementation from previous consolidated file) ...
    print(f"[Coordinator Tool] Decomposing query: '{user_query[:100]}...'")
    prompt = f"""You are an expert research assistant... decompose... Output *only* a Python list of strings. Example Input: "How did COVID-19 impact..." Example Output: ["Identify economic impacts...", ...] User Query: "{user_query}" Decomposed Sub-queries (Python list of strings):"""
    try:
        response = coordinator_llm.invoke([SystemMessage(content=prompt)])
        raw_list_str = response.content.strip()
        if raw_list_str.startswith('[') and raw_list_str.endswith(']'):
             import ast
             sub_tasks = ast.literal_eval(raw_list_str)
             if isinstance(sub_tasks, list) and all(isinstance(item, str) for item in sub_tasks): return sub_tasks
             else: raise ValueError("Invalid list")
        else: return [raw_list_str]
    except Exception as e: return [f"Address original query: {user_query}"]

@coordinator_mcp.tool()
def result_synthesis_tool(original_query: str, all_gathered_data: Dict[str, Any]) -> str:
    # ... (Implementation from previous consolidated file) ...
    print(f"[Coordinator Tool] Synthesizing results for: '{original_query[:100]}...'")
    data_summary = "\n\n## Research Findings:\n"
    for task, results_list in all_gathered_data.items():
        data_summary += f"\n### Sub-Task: {task}\n"
        if isinstance(results_list, list):
            for result_item in results_list:
                tool_name = result_item.get("tool_called", "unknown")
                tool_result = result_item.get("tool_result", "N/A")
                res_str = str(tool_result); res_str = (res_str[:300] + "...") if len(res_str) > 300 else res_str
                data_summary += f"- **{tool_name}**: {res_str}\n"
        else: res_str = str(results_list); res_str = (res_str[:300] + "...") if len(res_str) > 300 else res_str; data_summary += f"- Findings: {res_str}\n"
    prompt = f"""You are a research report writer... synthesize... Original User Query: "{original_query}" Instructions: Address query. Integrate findings. Cite sources. Incorporate confidence. Structure clearly. Be concise. Use *only* provided findings. Provided Findings:{data_summary}\n---\nFinal Report:"""
    try: return coordinator_llm.invoke([SystemMessage(content=prompt)]).content
    except Exception as e: return f"Error synthesizing report: {e}\n\nRaw Findings:\n{data_summary}"

# --- Agent 2 Tools (Use @rag_mcp.tool()) ---
@rag_mcp.tool()
def web_scrape_and_ingest(urls: List[str] | None = None) -> str:
    # ... (Implementation from previous semantic chunking file) ...
    global vector_store, embeddings_model
    print("[RAG Tool] WebScrape (Semantic Chunking)")
    urls_to_scrape = urls if urls else PREDEFINED_URL_LIST; print(f"Scraping {len(urls_to_scrape)} URLs...")
    loader = WebBaseLoader(web_paths=urls_to_scrape)
    try: docs_list = loader.load()
    except Exception as e: return f"Error loading URLs: {e}"
    if not docs_list: return "Error: No documents loaded."
    text_splitter = SemanticChunker(embeddings_model, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95)
    chunks = []
    for doc in docs_list:
         semantic_chunks_texts = text_splitter.split_text(doc.page_content)
         for i, chunk_text in enumerate(semantic_chunks_texts):
             chunk_metadata = doc.metadata.copy(); chunk_metadata["semantic_chunk_index"] = i
             chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
    print(f"Total semantic chunks: {len(chunks)}.")
    if not chunks: return "No text after chunking."
    for chunk in chunks:
        if len(chunk.page_content.split()) > 5:
            keywords_tuples = kw_extractor.extract_keywords(chunk.page_content)
            chunk.metadata["keywords"] = [kw[0].lower() for kw in keywords_tuples]
        else: chunk.metadata["keywords"] = []
    try: vector_store.add_documents(chunks)
    except Exception as e: return f"Error: DB ingestion failed. {e}"
    return f"Success: Scraped/chunked {len(urls_to_scrape)} URLs. Added {len(chunks)} chunks."

@rag_mcp.tool()
def semantic_search_tool(query: str, k: int = 4) -> List[Dict[str, Any]]:
    # ... (Implementation from previous semantic chunking file) ...
    global vector_store
    print(f"[RAG Tool] Semantic search: '{query}'")
    try:
        results = vector_store.similarity_search_with_score(query, k=k)
        formatted = [{"text_content": doc.page_content, "source": doc.metadata.get('source', 'N/A'), "similarity_score": score} for doc, score in results]
        return formatted if formatted else [{"status": "No relevant documents found."}]
    except Exception as e: return [{"error": f"Error during semantic search: {e}"}]

@rag_mcp.tool()
def keyword_search_tool(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    # ... (Implementation from previous semantic chunking file) ...
    global db_engine, vector_store
    print(f"[RAG Tool] Keyword search: '{query}'")
    try:
        table_name = vector_store.embedding_table_name
        query_keywords = list(set(query.lower().split()))
        sql = text(f"SELECT document, cmetadata FROM {table_name} WHERE cmetadata -> 'keywords' ?| array[:query_keywords] LIMIT :limit")
        formatted = []
        with db_engine.connect() as conn:
            res = conn.execute(sql, {"query_keywords": query_keywords, "limit": limit})
            for row in res.mappings():
                formatted.append({"text_content": row['document'], "source": row['cmetadata'].get('source', 'N/A'), "matching_keywords": [kw for kw in row['cmetadata'].get('keywords', []) if kw in query_keywords]})
        return formatted if formatted else [{"status": "No relevant documents found."}]
    except Exception as e: return [{"error": f"Error during keyword search: {e}"}]


# --- Agent 3 Tools (Use @analysis_mcp.tool()) ---
def _internal_semantic_search_a3(query: str, k: int = 3) -> List[Dict[str, Any]]: # Renamed helper
    # ... (Implementation from previous agent 3 file) ...
    global vector_store; print(f"[Agent 3 Internal] Semantic search: '{query}' (k={k})")
    try: results = vector_store.similarity_search_with_score(query, k=k); return [{"source": doc.metadata.get('source', 'N/A'), "text": doc.page_content} for doc, score in results]
    except Exception as e: return [{"source": "Error", "text": f"Error retrieving context: {e}"}]

@analysis_mcp.tool()
def comparative_analysis_tool(topic1: str, topic2: str) -> str:
    # ... (Implementation from previous agent 3 file, using _internal_semantic_search_a3) ...
    print(f"[Analysis Tool] Comparative Analysis: '{topic1}' vs '{topic2}'")
    context1_list = _internal_semantic_search_a3(topic1); context2_list = _internal_semantic_search_a3(topic2)
    context1 = "\n---\n".join([c['text'] for c in context1_list if 'Error' not in c['source']]); context2 = "\n---\n".join([c['text'] for c in context2_list if 'Error' not in c['source']])
    combined = f"Ctx '{topic1}':\n{context1}\n\nCtx '{topic2}':\n{context2}"; prompt = f"Compare/contrast '{topic1}' vs '{topic2}' based ONLY on context. Focus diffs if contradictory, else nuances. Structure clearly. No outside knowledge. Context:\n---\n{combined}\n---\nAnalysis:"
    try: return analysis_llm.invoke([SystemMessage(content=prompt)]).content
    except Exception as e: return f"Error: {e}"

@analysis_mcp.tool()
def trend_analysis_tool(topic: str) -> str:
    # ... (Implementation from previous agent 3 file, using _internal_semantic_search_a3) ...
    print(f"[Analysis Tool] Trend Analysis: '{topic}'")
    context_list = _internal_semantic_search_a3(topic, k=10); context = "\n---\n".join([c['text'] for c in context_list if 'Error' not in c['source']])
    prompt = f"Identify temporal trends for '{topic}' based ONLY on context. Look for changes over time/dates/sequences. If none, state clearly. No outside knowledge. Context:\n---\n{context}\n---\nTrend Analysis:"
    try: return analysis_llm.invoke([SystemMessage(content=prompt)]).content
    except Exception as e: return f"Error: {e}"

@analysis_mcp.tool()
def causal_reasoning_tool(effect_or_event: str, potential_causes_query: str) -> str:
    # ... (Implementation from previous agent 3 file, using _internal_semantic_search_a3) ...
    print(f"[Analysis Tool] Causal Reasoning: '{effect_or_event}'")
    context_list = _internal_semantic_search_a3(f"{effect_or_event} and {potential_causes_query}", k=5); context = "\n---\n".join([c['text'] for c in context_list if 'Error' not in c['source']])
    prompt = f"Identify cause-effect links for '{effect_or_event}' based ONLY on context. State causes/effects. If none, state that. No outside knowledge. Context:\n---\n{context}\n---\nCausal Analysis:"
    try: return analysis_llm.invoke([SystemMessage(content=prompt)]).content
    except Exception as e: return f"Error: {e}"

@analysis_mcp.tool()
def statistical_analysis_tool(quantitative_query: str) -> str:
    # ... (Implementation from previous agent 3 file, using _internal_semantic_search_a3) ...
    print(f"[Analysis Tool] Statistical Analysis: '{quantitative_query}'")
    context_list = _internal_semantic_search_a3(quantitative_query, k=5); context = "\n---\n".join([c['text'] for c in context_list if 'Error' not in c['source']])
    prompt = f'Answer: "{quantitative_query}". Extract numbers/facts ONLY from context. Calculate ONLY from data. Format clearly. If data unavailable, state that. No outside knowledge. Context:\n---\n{context}\n---\nStatistical Answer:'
    try: return analysis_llm.invoke([SystemMessage(content=prompt)]).content
    except Exception as e: return f"Error: {e}"


# --- Agent 4 Tools (Use @validation_mcp.tool()) ---
# Helper needed by Agent 4
def _normalize_domain_a4(url: str) -> str:
    # ... (Implementation from previous agent 4 file) ...
    try: match = re.search(r"^(?:https?:\/\/)?(?:[^@\n]+@)?(?:www\.)?([^:\/\n?]+)", url); return match.group(1) if match else url.split('/')[0] if '/' in url and '.' in url.split('/')[0] else url
    except Exception: return url

@validation_mcp.tool()
def source_credibility_checker(source_url: str) -> Dict[str, Any]:
    # ... (Implementation from previous agent 4 OS file) ...
    global TRUSTED_SOURCE_LIST; print(f"[Validation Tool] Credibility Check: '{source_url}'")
    domain = _normalize_domain_a4(source_url); justification = []; score = 5
    if domain in TRUSTED_SOURCE_LIST: return TRUSTED_SOURCE_LIST[domain]
    url_to_check = source_url if source_url.startswith('http') else 'https://' + source_url
    try: response = requests.get(url_to_check, timeout=5, allow_redirects=True, verify=True); score += 1 if response.url.startswith("https://") else -2; justification.append("Uses HTTPS." if response.url.startswith("https://") else "No HTTPS.")
    except requests.exceptions.SSLError: score -= 3; justification.append("HTTPS cert failed.")
    except requests.exceptions.RequestException as e: score -= 1; justification.append(f"No HTTPS connect ({type(e).__name__}).")
    try: whois_domain = _normalize_domain_a4(url_to_check); domain_info = whois.whois(whois_domain); creation_date = domain_info.creation_date; creation_date = creation_date[0] if isinstance(creation_date, list) else creation_date
        if isinstance(creation_date, datetime.datetime): now = datetime.datetime.now(creation_date.tzinfo); age = (now - creation_date).days / 365.25; justification.append(f"Age {age:.1f} yrs."); score += 2 if age > 5 else (1 if age > 1 else -1)
        else: justification.append("Date unclear."); score -=1
    except whois.parser.PywhoisError: justification.append(f"No WHOIS."); score -= 1
    except Exception as e: justification.append(f"WHOIS Error: {type(e).__name__}"); score -= 1
    final_score = max(1, min(score, 10)); final_just = " ".join(justification) if justification else "Checks inconclusive."
    return {"score": final_score, "justification": final_just}

@validation_mcp.tool()
def cross_reference_validator(claim: str, query_for_context: str) -> Dict[str, Any]:
    # ... (Implementation from previous agent 4 self-sufficient file) ...
    print(f"[Validation Tool] Cross-Reference: '{claim}' using ctx for '{query_for_context}'")
    contexts = _internal_semantic_search_a3(query_for_context, k=5) # Use Agent 3's helper
    if not contexts or "Error" in contexts[0]["source"]: return {"claim": claim, "error": "Could not retrieve context."}
    internal = {"supporting": [], "refuting": [], "neutral": []}
    for i, item in enumerate(contexts): text = item.get('text', ''); src = item.get('source', f'Ctx{i+1}')
        prompt = f'NLI: Claim: "{claim}" Ctx: "{text}" Based *only* on ctx, does it: SUPPORT, REFUTE, or NEUTRAL? Single word.'
        try:
            if not text or not text.strip(): internal["neutral"].append(src + " (Empty)"); continue
            res = heavy_llm.invoke([SystemMessage(content=prompt)]).content.upper()
            if "SUPPORT" in res: internal["supporting"].append(src)
            elif "REFUTE" in res: internal["refuting"].append(src)
            else: internal["neutral"].append(src)
        except Exception as e: internal["neutral"].append(src + f" (NLI Error: {e})")
    return {"claim": claim, "internal_validation": internal}

@validation_mcp.tool()
def contradiction_detector(query_for_context: str, num_docs_to_check: int = 5) -> List[Dict[str, Any]]:
    # ... (Implementation from previous agent 4 self-sufficient file) ...
    k = max(2, min(num_docs_to_check, 10)); print(f"[Validation Tool] Contradiction Detect: ctx for '{query_for_context}' (top {k})")
    contexts = _internal_semantic_search_a3(query_for_context, k=k); contradictions = []
    if not contexts or "Error" in contexts[0]["source"] or len(contexts) < 2: return []
    for (item1, item2) in itertools.combinations(contexts, 2):
        text1 = item1.get('text', ''); text2 = item2.get('text', ''); src1 = item1.get('source', 'Src1'); src2 = item2.get('source', 'Src2')
        if not text1 or not text2 or not text1.strip() or not text2.strip(): continue
        prompt = f'Contradiction check: A (from {src1}): "{text1}" B (from {src2}): "{text2}" Do they directly contradict? Answer "YES" or "NO", with explanation.'
        try:
            res = heavy_llm.invoke([SystemMessage(content=prompt)]).content
            if res.strip().upper().startswith("YES"): contradictions.append({"sources": [src1, src2], "explanation": res})
        except Exception as e: print(f"Error checking contradiction {src1} vs {src2}: {e}")
    return contradictions

@validation_mcp.tool()
def confidence_scorer(claim: str, cross_ref_results: Dict[str, Any], contradiction_results: List[Dict[str, Any]], credibility_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    # ... (Implementation from previous agent 4 OS file) ...
    print(f"[Validation Tool] Confidence Score for: '{claim}'")
    internal_val = cross_ref_results.get('internal_validation', {}); sup = internal_val.get('supporting', []); ref = internal_val.get('refuting', []); neu = internal_val.get('neutral', [])
    summary = f"""Claim: "{claim}" Report: 1. Internal XRef: Sup={sup}, Ref={ref}, Neu={neu}. 2. Contradictions: {len(contradiction_results)}. {'Details:' + str(contradiction_results) if contradiction_results else ''}. 3. Credibility: {credibility_reports}"""
    prompt = f"""Assign confidence (0-100) based ONLY on report. Guidelines: High internal sup (cred>=7, no contr) = 80+. Internal ref (cred>=7) = <20. Contr between cred sources = max 50. Sup only low cred (<=4) = max 40. Mixed/medium cred (5-6) = 40-60. Lack evidence = 10-30. Report:\n{summary}\nFormat:\nScore: [score]\nJustification: [justification...]"""
    try:
        res = fast_llm.invoke([SystemMessage(content=prompt)]).content
        score = int(re.search(r"Score: (\d+)", res).group(1)) if re.search(r"Score: (\d+)", res) else 0
        just = re.search(r"Justification: ([\s\S]*)", res).group(1).strip() if re.search(r"Justification: ([\s\S]*)", res) else "N/A."
        score = max(0, min(score, 100)); return {"score": score, "justification": just}
    except Exception as e: return {"score": 0, "justification": f"Error scoring: {e}"}

# --- 5. RUN THE SERVER (Single mcp.run) ---
if __name__ == "__main__":
    print("[All Tools Server] Starting consolidated server on default port (likely 8000)...")
    # --- Try running one instance; hope it picks up others ---
    # We choose coordinator_mcp arbitrarily. This might only expose Coordinator_Tools.
    # The success depends on internal FastMCP registration behavior.
    try:
        coordinator_mcp.run(transport="streamable-http")
        # If the above line runs without accepting host/port and only exposes one set of tools,
        # this consolidated approach with a single mcp.run might not work as intended for HTTP.
        # The FastAPI mounting approach is generally more reliable for multiple namespaces over HTTP.
    except TypeError as e:
        print("\n--- ERROR ---")
        print(f"Failed to run mcp.run: {e}")
        print("This confirms mcp.run() might not accept port/host directly as expected.")
        print("Consider using the FastAPI + uvicorn approach from the previous example for reliable multi-namespace HTTP serving.")
        print("Or, run each agent's tools in a separate process/terminal, each using mcp.run() on a different port.")
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR during server run --- : {e}")
    
