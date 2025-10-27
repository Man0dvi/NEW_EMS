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

# LangChain components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text
from langchain_core.documents import Document

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
    """ Breaks down complex user query into sub-tasks. """
    print(f"[Coordinator Tool] Decomposing query: '{user_query[:50]}...'")
    prompt = f"""Decompose into 3-6 Python list items: "{user_query}" Example: ["Task 1", "Task 2"] Output ONLY the list."""
    try:
        response = coordinator_llm.invoke([SystemMessage(content=prompt)])
        raw_list_str = response.content.strip()
        if raw_list_str.startswith('[') and raw_list_str.endswith(']'):
             import ast
             sub_tasks = ast.literal_eval(raw_list_str)
             if isinstance(sub_tasks, list) and all(isinstance(item, str) for item in sub_tasks): return sub_tasks
        return [raw_list_str] # Fallback
    except Exception as e: print(f"Decomp Error: {e}"); return [f"Original: {user_query}"]

@coordinator_mcp.tool()
def result_synthesis_tool(original_query: str, all_gathered_data: Dict[str, Any]) -> str:
    """ Synthesizes final answer from gathered data. """
    print(f"[Coordinator Tool] Synthesizing for: '{original_query[:50]}...'")
    data_summary = "\n## Findings:\n"
    for task, results_list in all_gathered_data.items():
        data_summary += f"\n### Task: {task}\n"
        if isinstance(results_list, list):
            for item in results_list:
                tool = item.get("tool_called", "?"); res = str(item.get("tool_result", "N/A"))
                data_summary += f"- {tool}: {(res[:200] + '...') if len(res) > 200 else res}\n"
        else: res = str(results_list); data_summary += f"- Findings: {(res[:200] + '...') if len(res) > 200 else res}\n"
    prompt = f"""Write a comprehensive report answering: "{original_query}". Use ONLY these findings. Cite sources/confidence if available. Structure clearly. Findings:{data_summary}\n---\nReport:"""
    try: return coordinator_llm.invoke([SystemMessage(content=prompt)]).content
    except Exception as e: return f"Synthesis Error: {e}\nRaw:{data_summary}"

# --- Agent 2 Tools (Use @rag_mcp.tool()) ---
@rag_mcp.tool()
def web_scrape_and_ingest(urls: List[str] | None = None) -> str:
    """ Tool 1 (Agent 2): Scrapes, chunks (semantic), adds keywords, ingests. """
    global vector_store, embeddings_model, kw_extractor, PREDEFINED_URL_LIST # Ensure access
    print("[RAG Tool] WebScrape (Semantic Chunking)")
    urls_to_scrape = urls if urls else PREDEFINED_URL_LIST; print(f"Scraping {len(urls_to_scrape)} URLs...")
    loader = WebBaseLoader(web_paths=urls_to_scrape)
    try: docs_list = loader.load()
    except Exception as e: return f"Load Error: {e}"
    if not docs_list: return "Error: No docs loaded."
    text_splitter = SemanticChunker(embeddings_model, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95)
    chunks = []
    for doc in docs_list:
         texts = text_splitter.split_text(doc.page_content)
         for i, text in enumerate(texts):
             meta = doc.metadata.copy(); meta["semantic_chunk_index"] = i
             chunks.append(Document(page_content=text, metadata=meta))
    print(f"Total chunks: {len(chunks)}.")
    if not chunks: return "No text after chunking."
    for chunk in chunks:
        if len(chunk.page_content.split()) > 5:
            kws = kw_extractor.extract_keywords(chunk.page_content)
            chunk.metadata["keywords"] = [kw[0].lower() for kw in kws]
        else: chunk.metadata["keywords"] = []
    try: vector_store.add_documents(chunks)
    except Exception as e: return f"DB Ingest Error: {e}"
    return f"Success: Scraped/chunked {len(urls_to_scrape)} URLs. Added {len(chunks)} chunks."

@rag_mcp.tool()
def semantic_search_tool(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """ Tool 2 (Agent 2): Semantic search. """
    global vector_store; print(f"[RAG Tool] Semantic search: '{query}'")
    try:
        results = vector_store.similarity_search_with_score(query, k=k)
        fmt = [{"text": d.page_content, "src": d.metadata.get('source','?'), "score": s} for d, s in results]
        return fmt if fmt else [{"status": "No results."}]
    except Exception as e: return [{"error": f"Semantic Search Error: {e}"}]

@rag_mcp.tool()
def keyword_search_tool(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """ Tool 3 (Agent 2): Keyword search on metadata. """
    global db_engine, vector_store; print(f"[RAG Tool] Keyword search: '{query}'")
    try:
        tbl = vector_store.embedding_table_name; kws = list(set(query.lower().split()))
        sql = text(f"SELECT document, cmetadata FROM {tbl} WHERE cmetadata->'keywords' ?| array[:kws] LIMIT :lim")
        fmt = []
        with db_engine.connect() as conn:
            res = conn.execute(sql, {"kws": kws, "lim": limit})
            for row in res.mappings():
                match_kws = [k for k in row['cmetadata'].get('keywords',[]) if k in kws]
                fmt.append({"text": row['document'], "src": row['cmetadata'].get('source','?'), "match_kws": match_kws})
        return fmt if fmt else [{"status": "No results."}]
    except Exception as e: return [{"error": f"Keyword Search Error: {e}"}]

# --- Agent 3 Tools (Use @analysis_mcp.tool()) ---
def _internal_semantic_search_a3(query: str, k: int = 3) -> List[Dict[str, Any]]: # Renamed
    global vector_store; print(f"[A3 Internal] Sem search: '{query}' (k={k})")
    try: results = vector_store.similarity_search_with_score(query, k=k); return [{"source": d.metadata.get('source','?'), "text": d.page_content} for d, s in results]
    except Exception as e: return [{"source": "Error", "text": f"Ctx Error: {e}"}]

@analysis_mcp.tool()
def comparative_analysis_tool(topic1: str, topic2: str) -> str:
    """ Tool 1 (Agent 3): Compares/contrasts topic1 and topic2 using retrieved context. """
    print(f"[Analysis Tool] Compare: '{topic1}' vs '{topic2}'")
    ctx1 = _internal_semantic_search_a3(topic1); ctx2 = _internal_semantic_search_a3(topic2)
    t1 = "\n---\n".join([c['text'] for c in ctx1 if 'Error' not in c['source']])
    t2 = "\n---\n".join([c['text'] for c in ctx2 if 'Error' not in c['source']])
    combo = f"Ctx '{topic1}':\n{t1}\n\nCtx '{topic2}':\n{t2}"; p = f"Compare/contrast '{topic1}' vs '{topic2}' based ONLY on context. Focus diffs if contradictory, else nuances. Structure. No outside knowledge. Context:\n---\n{combo}\n---\nAnalysis:"
    try: return analysis_llm.invoke([SystemMessage(content=p)]).content
    except Exception as e: return f"Compare Error: {e}"

@analysis_mcp.tool()
def trend_analysis_tool(topic: str) -> str:
    """ Tool 2 (Agent 3): Analyzes temporal trends for topic using retrieved context. """
    print(f"[Analysis Tool] Trend: '{topic}'")
    ctx_list = _internal_semantic_search_a3(topic, k=10); ctx = "\n---\n".join([c['text'] for c in ctx_list if 'Error' not in c['source']])
    p = f"Identify temporal trends for '{topic}' ONLY from context. Look for dates/sequences/changes. If none, state clearly. No outside knowledge. Context:\n---\n{ctx}\n---\nTrend Analysis:"
    try: return analysis_llm.invoke([SystemMessage(content=p)]).content
    except Exception as e: return f"Trend Error: {e}"

@analysis_mcp.tool()
def causal_reasoning_tool(effect_or_event: str, potential_causes_query: str) -> str:
    """ Tool 3 (Agent 3): Identifies cause-effect relationships using retrieved context. """
    print(f"[Analysis Tool] Causal: '{effect_or_event}'")
    ctx_list = _internal_semantic_search_a3(f"{effect_or_event} {potential_causes_query}", k=5); ctx = "\n---\n".join([c['text'] for c in ctx_list if 'Error' not in c['source']])
    p = f"Identify cause-effect for '{effect_or_event}' ONLY from context. State causes/effects. If none, state that. No outside knowledge. Context:\n---\n{ctx}\n---\nCausal Analysis:"
    try: return analysis_llm.invoke([SystemMessage(content=p)]).content
    except Exception as e: return f"Causal Error: {e}"

@analysis_mcp.tool()
def statistical_analysis_tool(quantitative_query: str) -> str:
    """ Tool 4 (Agent 3): Extracts/calculates quantitative data using retrieved context. """
    print(f"[Analysis Tool] Stats: '{quantitative_query}'")
    ctx_list = _internal_semantic_search_a3(quantitative_query, k=5); ctx = "\n---\n".join([c['text'] for c in ctx_list if 'Error' not in c['source']])
    p = f'Answer: "{quantitative_query}". Extract numbers/facts ONLY from context. Calculate ONLY from data. Format clearly. If data unavailable, state that. No outside knowledge. Context:\n---\n{ctx}\n---\nStatistical Answer:'
    try: return analysis_llm.invoke([SystemMessage(content=p)]).content
    except Exception as e: return f"Stats Error: {e}"


# --- Agent 4 Tools (Use @validation_mcp.tool()) ---
def _normalize_domain_a4(url: str) -> str: # Agent 4 Helper
    try: match = re.search(r"^(?:https?:\/\/)?(?:[^@\n]+@)?(?:www\.)?([^:\/\n?]+)", url); return match.group(1) if match else url.split('/')[0] if '/' in url and '.' in url.split('/')[0] else url
    except: return url

@validation_mcp.tool()
def source_credibility_checker(source_url: str) -> Dict[str, Any]:
    """ Tool 1 (Agent 4): Evaluates source reliability (OS version). """
    global TRUSTED_SOURCE_LIST; print(f"[Validation Tool] Credibility: '{source_url}'")
    domain = _normalize_domain_a4(source_url); just = []; score = 5
    if domain in TRUSTED_SOURCE_LIST: return TRUSTED_SOURCE_LIST[domain]
    url = source_url if source_url.startswith('http') else 'https://'+source_url
    try: r = requests.get(url, timeout=5, allow_redirects=True, verify=True); score += 1 if r.url.startswith("https://") else -2; just.append("HTTPS." if r.url.startswith("https://") else "No HTTPS.")
    except requests.exceptions.SSLError: score -= 3; just.append("HTTPS cert failed.")
    except requests.exceptions.RequestException as e: score -= 1; just.append(f"No HTTPS check ({type(e).__name__}).")
    try: d_info = whois.whois(_normalize_domain_a4(url)); cd = d_info.creation_date; cd = cd[0] if isinstance(cd, list) else cd
        if isinstance(cd, datetime.datetime): age = (datetime.datetime.now(cd.tzinfo) - cd).days/365.25; just.append(f"Age {age:.1f}y."); score += 2 if age > 5 else (1 if age > 1 else -1)
        else: just.append("Date unclear."); score -=1
    except whois.parser.PywhoisError: just.append(f"No WHOIS."); score -= 1
    except Exception as e: just.append(f"WHOIS Err: {type(e).__name__}"); score -= 1
    fs = max(1, min(score, 10)); fj = " ".join(just) if just else "Checks inconclusive."
    return {"score": fs, "justification": fj}

@validation_mcp.tool()
def cross_reference_validator(claim: str, query_for_context: str) -> Dict[str, Any]:
    """ Tool 2 (Agent 4): Retrieves context, verifies claim internally via NLI. """
    print(f"[Validation Tool] Cross-Ref: '{claim}' using ctx for '{query_for_context}'")
    contexts = _internal_semantic_search_a3(query_for_context, k=5)
    if not contexts or "Error" in contexts[0].get("source",""): return {"claim": claim, "error": "Ctx retrieval failed."}
    internal = {"supporting": [], "refuting": [], "neutral": []}
    for i, item in enumerate(contexts): text = item.get('text', ''); src = item.get('source', f'Ctx{i+1}')
        prompt = f'NLI: Claim="{claim}" Ctx="{text}" Based ONLY on ctx: SUPPORT, REFUTE, or NEUTRAL? Single word.'
        try:
            if not text or not text.strip(): internal["neutral"].append(src + "(Empty)"); continue
            res = heavy_llm.invoke([SystemMessage(content=prompt)]).content.upper()
            if "SUPPORT" in res: internal["supporting"].append(src)
            elif "REFUTE" in res: internal["refuting"].append(src)
            else: internal["neutral"].append(src)
        except Exception as e: internal["neutral"].append(src + f"(NLI Err:{e})")
    return {"claim": claim, "internal_validation": internal}

@validation_mcp.tool()
def contradiction_detector(query_for_context: str, num_docs_to_check: int = 5) -> List[Dict[str, Any]]:
    """ Tool 3 (Agent 4): Retrieves context, finds internal contradictions. """
    k = max(2, min(num_docs_to_check, 10)); print(f"[Validation Tool] Contradiction: ctx for '{query_for_context}' (top {k})")
    contexts = _internal_semantic_search_a3(query_for_context, k=k); contradictions = []
    if not contexts or "Error" in contexts[0].get("source","") or len(contexts) < 2: return []
    for (item1, item2) in itertools.combinations(contexts, 2):
        t1=item1.get('text',''); t2=item2.get('text',''); s1=item1.get('source','S1'); s2=item2.get('source','S2')
        if not t1 or not t2 or not t1.strip() or not t2.strip(): continue
        prompt = f'Contradiction check: A(from {s1}):"{t1}" B(from {s2}):"{t2}" Do they directly contradict? YES/NO + explanation.'
        try: res = heavy_llm.invoke([SystemMessage(content=prompt)]).content
            if res.strip().upper().startswith("YES"): contradictions.append({"sources": [s1, s2], "explanation": res})
        except Exception as e: print(f"Contradiction check err {s1} vs {s2}: {e}")
    return contradictions

@validation_mcp.tool()
def confidence_scorer(claim: str, cross_ref_results: Dict[str, Any], contradiction_results: List[Dict[str, Any]], credibility_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """ Tool 4 (Agent 4): Assigns final confidence score based on evidence. """
    print(f"[Validation Tool] Confidence Score: '{claim}'")
    iv = cross_ref_results.get('internal_validation',{}); s=iv.get('supporting',[]); r=iv.get('refuting',[]); n=iv.get('neutral',[])
    summ = f"""Claim:"{claim}" Report: 1.Internal XRef: Sup={s}, Ref={r}, Neu={n}. 2.Contradictions: {len(contradiction_results)}. {'Details:'+str(contradiction_results) if contradiction_results else ''}. 3.Credibility:{credibility_reports}"""
    p = f"""Assign confidence (0-100) based ONLY on report. Guidelines: High internal sup(cred>=7,no contr)=80+. Internal ref(cred>=7)=<20. Contr between cred sources=max 50. Sup only low cred(<=4)=max 40. Mixed/medium cred(5-6)=40-60. Lack evidence=10-30. Report:\n{summ}\nFormat:\nScore: [score]\nJustification: [justification...]"""
    try:
        res = fast_llm.invoke([SystemMessage(content=p)]).content; score_m = re.search(r"Score: (\d+)", res); just_m = re.search(r"Justification: ([\s\S]*)", res)
        score = int(score_m.group(1)) if score_m else 0; just = just_m.group(1).strip() if just_m else "N/A."
        score = max(0, min(score, 100)); return {"score": score, "justification": just}
    except Exception as e: return {"score": 0, "justification": f"Error scoring: {e}"}

# --- 5. RUN THE SERVER (Single mcp.run on default port) ---
if __name__ == "__main__":
    print("[All Tools Server] Starting consolidated server on default port (likely 8000)...")
    try:
        # Run using one of the MCP instances (e.g., coordinator_mcp)
        # This relies on the library correctly registering tools from all defined instances.
        coordinator_mcp.run(transport="streamable-http")
    except TypeError as e:
        print(f"\n--- ERROR --- : Failed to run mcp.run: {e}")
        print("This suggests mcp.run() might not correctly handle multiple instances this way for HTTP.")
        print("Consider using the FastAPI + uvicorn approach with app.mount() for reliable multi-namespace HTTP serving.")
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR during server run --- : {e}")
