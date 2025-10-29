from pydantic import BaseModel, Field
from typing import List, Optional

class AnalysisConfig(BaseModel):
    project_id: str
    personas: List[str] = Field(default_factory=list)
    analysis_depth: str = Field(default="quick")      # quick | standard | deep
    verbosity: str = Field(default="standard")        # low | standard | high
    enable_web_search: bool = False
    diagram_enabled: bool = False
    include_files: Optional[List[str]] = None
    exclude_files: Optional[List[str]] = None
