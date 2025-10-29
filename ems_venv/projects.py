# projects_router.py (new)
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.db import get_db
from models.model import Project  # adapt to your model
from pydantic import BaseModel

router = APIRouter(prefix="/projects", tags=["projects"])

class ProjectIn(BaseModel):
    name: str
    personas: str = "SDE,PM"

@router.get("")
def list_projects(db: Session = Depends(get_db)):
    return db.query(Project).order_by(Project.id.desc()).all()

@router.get("/{pid}")
def get_project(pid: int, db: Session = Depends(get_db)):
    p = db.get(Project, pid)
    if not p:
        raise HTTPException(404, "Not found")
    return p

@router.post("")
def create_project(payload: ProjectIn, db: Session = Depends(get_db)):
    p = Project(name=payload.name, personas=payload.personas, status="created")
    db.add(p); db.commit(); db.refresh(p)
    return p

@router.patch("/{pid}")
def update_project(pid: int, payload: ProjectIn, db: Session = Depends(get_db)):
    p = db.get(Project, pid)
    if not p:
        raise HTTPException(404, "Not found")
    p.name = payload.name
    p.personas = payload.personas
    db.commit()
    return p

@router.delete("/{pid}")
def delete_project(pid: int, db: Session = Depends(get_db)):
    p = db.get(Project, pid)
    if not p:
        raise HTTPException(404, "Not found")
    db.delete(p); db.commit()
    return {"ok": True}
