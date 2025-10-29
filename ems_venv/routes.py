from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from analysis_config import AnalysisConfig
from auth import get_current_user  # Existing token-based function
from model import CodeAnalysisState
from graph_deep import build_deep_analysis_graph

router = APIRouter()

@router.post("/start-deep-analysis/", response_model=Dict[str, Any])
async def start_deep_analysis(config: AnalysisConfig, user=Depends(get_current_user)):
    # Persona/authorization check
    if not set(config.personas).issubset(set(user.personas)):
        raise HTTPException(status_code=403, detail="You are not authorized to run analysis for some personas.")

    # Project path resolution (adapt to your repo organization)
    project_path = f"/data/projects/{config.project_id}"
    # Optionally check project path/file existence here as needed

    # Enrich config with resolved project path
    graph_config = config.dict()
    graph_config['project_path'] = project_path

    # Build and invoke the agent graph (sync for simplicity; async for bigger workloads)
    graph = build_deep_analysis_graph(CodeAnalysisState)
    initial_state = {**graph_config}
    final_state = await graph.ainvoke(initial_state)

    # Return the state for LLM QA or downstream consumption
    return {"status": "success", "state": final_state}
