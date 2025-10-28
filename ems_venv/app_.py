# --- crud/crud.py ---

from sqlalchemy.orm import Session
from models import model # Assuming models.py contains User, Project, Initialisation
from schema import schemas # Assuming schemas.py contains UserCreate etc.
from typing import Optional

# --- Existing User Functions ---
def get_user_by_email(db: Session, email: str) -> Optional[model.User]:
    return db.query(model.User).filter(model.User.email == email).first()

def create_user(db: Session, name: str, email: str, hashed_password: str) -> model.User:
    db_user = model.User(name=name, email=email, password=hashed_password) # Removed role assumption
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# --- Project Functions ---
def create_project(db: Session, user_id: int, project_name: str, file_name: str, local_path: str) -> model.Project:
    """ Creates a new project record, now including local_path. """
    db_project = model.Project(
        user_id=user_id,
        project_name=project_name,
        file_name=file_name,
        local_path=local_path # Add path here
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

# --- NEW: Get Project ---
def get_project_by_id(db: Session, project_id: int) -> Optional[model.Project]:
    """ Retrieves a project by its ID. """
    return db.query(model.Project).filter(model.Project.id == project_id).first()

# --- Initialisation Functions ---
# (Keep your existing create_initialization function)
def create_initialization(db: Session, user_id: int, project_id: int, persona: str) -> model.Initialisation:
    init = model.Initialisation(
        user_id=user_id,
        project_id=project_id,
        persona=persona,
        status="started" # Initial status
    )
    db.add(init)
    db.commit()
    db.refresh(init)
    return init

# --- NEW: Update Status ---
def update_initialization_status(db: Session, init_id: int, status: str, error_msg: Optional[str] = None):
    """ Updates the status and optionally error message of an Initialisation record. """
    init = db.query(model.Initialisation).filter(model.Initialisation.id == init_id).first()
    if init:
        init.status = status
        init.error_message = error_msg
        # updated_at should update automatically via onupdate in model
        db.commit()
        db.refresh(init)
    else:
        # Log or handle case where init record is not found
        print(f"Error: Could not find Initialisation with id {init_id} to update status.")

# --- NEW: Store Results ---
def store_analysis_result(db: Session, init_id: int, result_json: str):
    """ Stores the JSON results string in the Initialisation record. """
    init = db.query(model.Initialisation).filter(model.Initialisation.id == init_id).first()
    if init:
        init.results = result_json
        # updated_at should update automatically
        db.commit()
        db.refresh(init)
    else:
        print(f"Error: Could not find Initialisation with id {init_id} to store results.")

# --- NEW: Get Initialization ---
def get_initialization_by_id(db: Session, init_id: int) -> Optional[model.Initialisation]:
     """ Retrieves an initialization record by ID. """
     return db.query(model.Initialisation).filter(model.Initialisation.id == init_id).first()
