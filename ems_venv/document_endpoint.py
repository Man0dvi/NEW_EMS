from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models.model import Initialisation, User
from your_dependencies import get_db, get_current_user
from your_agents import WebSearchAgent, LLMFlowDiagramAgent
from your_pdf_utils import create_pdf_from_content  # Utility you will create to generate PDF
import json
import os
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/generate-documentation")
async def generate_documentation(
    user_id: int,
    project_id: int,
    config: dict = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized user")

    init_record = (
        db.query(Initialisation)
        .filter(Initialisation.user_id == user_id, Initialisation.project_id == project_id)
        .order_by(Initialisation.created_at.desc())
        .first()
    )

    if not init_record or not init_record.results:
        raise HTTPException(status_code=404, detail="No analysis results found for this project")

    final_state = json.loads(init_record.results)
    keywords = final_state.get("tech_stack") or []

    # Initialize state with final analysis data and keywords
    state = {
        **final_state,
        "keywords": keywords,
    }
    config = config or {}

    # Run web search agent
    web_agent = WebSearchAgent()
    state = await web_agent.process(state, config)

    # Run LLM-based Mermaid diagram agent
    llm_flow_agent = LLMFlowDiagramAgent()
    state = await llm_flow_agent.process(state, config)

    # Compile final document content (text + search results + mermaid diagram)
    doc_content = f"""
    Project ID: {project_id}

    Tech Stack Keywords:
    {', '.join(keywords)}

    Web Search Results:
    {state.get('web_search_results', 'No results')}

    Architecture Flow Diagram (Mermaid source):
    ```
    {state.get('mermaid_diagram', 'No diagram generated')}
    ```
    """

    # Generate PDF using your PDF utility function
    pdf_path = f"/tmp/project_{project_id}_documentation.pdf"
    create_pdf_from_content(doc_content, pdf_path)

    logger.info(f"Generated PDF documentation at {pdf_path}")

    return {
        "message": "Documentation generated successfully",
        "pdf_path": pdf_path,
        "project_id": project_id,
        "initialisation_id": init_record.id,
    }
