# admin_analytics_router.py (new)
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.db import get_db
from models.model import User, Project, Initialisation  # adapt to your names

router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/analytics")
def analytics(db: Session = Depends(get_db)):
    total_users = db.query(User).count()
    total_projects = db.query(Project).count()
    active_analyses = db.query(Initialisation).filter(Initialisation.status.in_(["started","processing"])).count()
    completion_rate = 0.0
    done = db.query(Initialisation).filter(Initialisation.status=="completed").count()
    total = db.query(Initialisation).count() or 1
    completion_rate = round(done/total*100, 2)
    return {
        "total_users": total_users,
        "total_projects": total_projects,
        "active_analyses": active_analyses,
        "completion_rate_pct": completion_rate
    }
