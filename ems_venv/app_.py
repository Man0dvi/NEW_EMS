# --- schema/schemas.py ---

from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

# --- Existing User Schemas ---
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr
    created_at: datetime

    class Config:
        orm_mode = True # For SQLAlchemy compatibility (Pydantic V1)
        # from_attributes = True # For Pydantic V2

# --- NEW: Initialization Output Schema ---
class InitializationOut(BaseModel):
    id: int
    project_id: int
    user_id: int
    persona: str
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    results: Optional[Dict[str, Any]] = None # Expect parsed JSON dictionary

    class Config:
        orm_mode = True # Pydantic V1
        # from_attributes = True # Pydantic V2

# --- You might also have Project schemas ---
# Example:
class ProjectOut(BaseModel):
    id: int
    user_id: int
    project_name: str
    file_name: str # URL or zip filename
    local_path: str
    created_at: datetime

    class Config:
        orm_mode = True
        # from_attributes = True
