# --- api/routes.py (Integrated Analysis Trigger) ---

import hashlib
import io
import os
import time # For unique folder names
from datetime import datetime, timedelta
import re
import zipfile
import json # For storing results
import logging
import asyncio # For running async graph

# FastAPI imports
from fastapi import (
    APIRouter, Depends, File, HTTPException, UploadFile, status, BackgroundTasks # Add BackgroundTasks
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Other imports
import git
from sqlalchemy.orm import Session
from jose import JWTError, jwt

# --- Your project imports (ADJUST PATHS if needed) ---
# Assuming 'multi_agent_docs' is accessible (e.g., in PYTHONPATH or installed)
# And assuming services, agents, tools, graph are within multi_agent_docs
try:
    # Import the function that builds and compiles the graph
    from multi_agent_docs.graph.graph import build_analysis_graph
    # Import the state definition for type hinting
    from multi_agent_docs.graph.graph import CodeAnalysisState
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    print("\n--- WARNING ---")
    print("Could not import graph components from 'multi_agent_docs'. Analysis endpoint will be disabled.")
    print(f"Import Error: {e}")
    print("Ensure 'multi_agent_docs' directory is in your Python path or installed correctly.\n")
    build_analysis_graph = None # Define as None if import fails
    CodeAnalysisState = dict # Use dict as fallback type
    ANALYSIS_AVAILABLE = False


# Database/Auth related imports
from models.model import Initialisation, User, Project # Make sure Project has local_path
from crud import crud # Import the whole module
# Ensure SessionLocal is defined in db.py for background tasks
from database.db import get_db, SessionLocal
from schema import schemas # Make sure schemas includes InitializationOut

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()
UPLOAD_BASE_DIR = os.path.abspath("uploads") # Use absolute path for reliability
os.makedirs(UPLOAD_BASE_DIR, exist_ok=True)

# --- JWT, Security, Regex, Constants (Keep as before) ---
SECRET_KEY = os.getenv("SECRET_KEY", "SECRET!ChangeMe!")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
MAX_ZIP_SIZE = 100 * 1024 * 1024
GIT_URL_REGEX = re.compile(r"^(https://|git@|ssh://)[\w\.\-/:]+\.git$")
ALLOWED_EXTENSIONS = {'.py', '.js', '.java', '.c', '.cpp', '.ts', '.go', '.rb', '.php', '.cs', '.rs'}

security = HTTPBearer()

# --- Helper Functions (create_access_token, get_current_user, has_code_files, is_repo_empty) ---
# --- (Keep these definitions as in your original file) ---
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials; credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})
    try: payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]); email: str = payload.get("sub");
        if email is None: raise credentials_exception
    except JWTError: raise credentials_exception
    user = crud.get_user_by_email(db, email);
    if user is None: raise credentials_exception
    return user

def has_code_files(repo_path):
    for root, dirs, files in os.walk(repo_path):
        # Exclude common non-code dirs early if possible (using EXCLUDE_DIRS from tools)
        # from multi_agent_docs.tools.repo_intel_tools import EXCLUDE_DIRS # Avoid import here if possible
        # dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS] # Optional optimization
        for file in files: _, ext = os.path.splitext(file);
            if ext.lower() in ALLOWED_EXTENSIONS: return True
    return False

def is_repo_empty(repo_path):
     for _, _, files in os.walk(repo_path):
        if files: return False
     return True

# --- User Endpoints (/users/, /token, /me - Keep as before) ---
@router.post("/users/", response_model=schemas.UserOut)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    if crud.get_user_by_email(db, user.email): raise HTTPException(status_code=400, detail="Email registered")
    hashed_pw = hashlib.sha256(user.password.encode()).hexdigest()
    new_user = crud.create_user(db, name=user.name, email=user.email, hashed_password=hashed_pw)
    return new_user

@router.post("/token")
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, user.email); hashed_pw = hashlib.sha256(user.password.encode()).hexdigest()
    if not db_user or db_user.password != hashed_pw: raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(data={"sub": db_user.email}); return {"access_token": token, "token_type": "bearer"}

@router.get("/me", response_model=schemas.UserOut)
def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user


# --- Background Task Function ---
# This needs to be defined BEFORE the endpoints that use it.
# It requires SessionLocal from database.db for creating a new DB session.
async def run_analysis_background(initialization_id: int, project_path: str, personas: List[str]):
    """ Compiles/runs LangGraph analysis, updates DB status/results. """
    db: Session = SessionLocal() # Create a new session for this background task
    graph_compiled = False
    analysis_graph = None # Define analysis_graph variable
    try:
        logger.info(f"[BG Task {initialization_id}] Starting analysis for path: {project_path}")
        # Ensure build_analysis_graph was imported successfully
        if build_analysis_graph is None:
             raise ImportError("Analysis graph module not loaded.")

        crud.update_initialization_status(db, initialization_id, "processing")

        # Compile graph (ensure build_analysis_graph is accessible)
        try:
             logger.info(f"[BG Task {initialization_id}] Compiling analysis graph...")
             # Assuming build_analysis_graph is synchronous after our refactor
             analysis_graph = build_analysis_graph()
             if not analysis_graph: raise RuntimeError("build_analysis_graph returned None.")
             graph_compiled = True
             logger.info(f"[BG Task {initialization_id}] Graph compiled successfully.")
        except Exception as compile_err:
             logger.error(f"[BG Task {initialization_id}] Graph compilation failed: {compile_err}", exc_info=True)
             crud.update_initialization_status(db, initialization_id, "failed", f"Graph compilation error: {compile_err}")
             return # Stop

        # Prepare initial state (ensure CodeAnalysisState is available)
        initial_state = CodeAnalysisState(project_path=project_path, personas=personas)
        config = {"recursion_limit": 15} # Adjust as needed

        # Run the graph asynchronously
        logger.info(f"[BG Task {initialization_id}] Invoking graph...")
        # Add timeout? Consider using asyncio.wait_for if needed
        # final_state = await asyncio.wait_for(analysis_graph.ainvoke(initial_state, config=config), timeout=600) # 10 min timeout example
        final_state = await analysis_graph.ainvoke(initial_state, config=config)
        logger.info(f"[BG Task {initialization_id}] Graph execution finished.")

        # Check for errors reported in the state
        final_status = "completed"
        error_msg = final_state.get("code_discovery_error") or \
                    final_state.get("file_proc_error") or \
                    final_state.get("repo_intel_error")

        if error_msg:
             final_status = "failed"
             logger.error(f"[BG Task {initialization_id}] Analysis failed in graph: {error_msg}")
             crud.update_initialization_status(db, initialization_id, final_status, str(error_msg))
        else:
             # Store results
             try:
                 # Clean up state slightly if needed
                 keys_to_store = ["tech_stack", "structure_summary", "architecture_summary", "complexity_assessment", "code_elements", "dependencies", "file_relationships", "suggested_skip_patterns", "semantic_chunks", "personas", "repo_files"]
                 result_to_store = {k: final_state.get(k) for k in keys_to_store if final_state.get(k) is not None} # Only store existing keys

                 # Use default=str for potential non-serializable types (like datetime in whois)
                 result_json = json.dumps(result_to_store, default=str)
                 crud.store_analysis_result(db, initialization_id, result_json)
                 crud.update_initialization_status(db, initialization_id, final_status)
                 logger.info(f"[BG Task {initialization_id}] Analysis completed and results stored.")
             except Exception as store_err:
                  logger.error(f"[BG Task {initialization_id}] Failed to store results: {store_err}", exc_info=True)
                  crud.update_initialization_status(db, initialization_id, "failed", f"Result storage error: {store_err}")

    except Exception as e:
        logger.error(f"[BG Task {initialization_id}] UNEXPECTED ERROR during analysis: {e}", exc_info=True)
        # Attempt to update status even on unexpected crash if possible
        try:
            # Check if init exists before updating
            init_exists = crud.get_initialization_by_id(db, initialization_id)
            if init_exists:
                 crud.update_initialization_status(db, initialization_id, "failed", f"Unexpected background error: {e}")
            else:
                 logger.error(f"[BG Task {initialization_id}] Initialization record not found, cannot update status after error.")
        except Exception as db_err:
             logger.error(f"[BG Task {initialization_id}] FAILED TO UPDATE STATUS AFTER ERROR: {db_err}")
    finally:
        db.close() # IMPORTANT: Close the background session


# --- Upload Endpoints (Modified for Background Task) ---
@router.post("/upload/", response_model=dict)
async def upload_zip(
    project_name: str,
    persona: str, # Should be List[str] maybe? For now single.
    zip_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = Depends()
):
    logger.info(f"Zip upload: P='{project_name}', User={current_user.id}")
    if not ANALYSIS_AVAILABLE: raise HTTPException(status_code=503, detail="Analysis service unavailable.")
    if persona not in ("SDE", "PM"): raise HTTPException(status_code=400, detail="Invalid persona.")
    if not zip_file.filename.lower().endswith(".zip"): raise HTTPException(status_code=400, detail="Must be .zip")

    contents = await zip_file.read()
    if len(contents) > MAX_ZIP_SIZE: raise HTTPException(status_code=413, detail=f"File > {MAX_ZIP_SIZE//1024//1024}MB.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_name = f"{project_name.replace(' ','_')}_{timestamp}"
    proj_path = os.path.abspath(os.path.join(UPLOAD_BASE_DIR, f"user_{current_user.id}", unique_name))

    try:
        os.makedirs(proj_path, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(contents), 'r') as zf: zf.extractall(proj_path)
        logger.info(f"Extracted zip to: {proj_path}")
        if is_repo_empty(proj_path): raise ValueError("Extracted ZIP is empty.")
        # if not has_code_files(proj_path): raise ValueError("No recognized code files found.") # Be less strict maybe?

    except (zipfile.BadZipFile, ValueError) as e:
        if os.path.exists(proj_path): import shutil; shutil.rmtree(proj_path)
        raise HTTPException(status_code=400, detail=f"Invalid/Empty ZIP: {e}")
    except Exception as e:
         logger.error(f"Zip extract error: {e}", exc_info=True)
         if os.path.exists(proj_path): import shutil; shutil.rmtree(proj_path) # Cleanup on failure
         raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    # DB entries
    project = crud.create_project(db, user_id=current_user.id, project_name=project_name, file_name=zip_file.filename, local_path=proj_path)
    init = crud.create_initialization(db, user_id=current_user.id, project_id=project.id, persona=persona)

    # Trigger background analysis
    background_tasks.add_task(run_analysis_background, init.id, proj_path, [persona])
    logger.info(f"Scheduled zip analysis: init_id={init.id}")

    return {"msg": f"Upload success. Analysis scheduled (Init ID: {init.id}).", "project_id": project.id, "initialization_id": init.id, "status": init.status}

@router.post("/upload-git/", response_model=dict)
async def upload_git_repo(
    project_name: str,
    persona: str, # Consider List[str] here too
    git_url: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = Depends()
):
    logger.info(f"Git upload: P='{project_name}', URL='{git_url}', User={current_user.id}")
    if not ANALYSIS_AVAILABLE: raise HTTPException(status_code=503, detail="Analysis service unavailable.")
    if persona not in ("SDE", "PM"): raise HTTPException(status_code=400, detail="Invalid persona.")
    if not GIT_URL_REGEX.match(git_url): raise HTTPException(status_code=400, detail="Invalid git URL.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_name = f"{project_name.replace(' ','_')}_{timestamp}"
    proj_path = os.path.abspath(os.path.join(UPLOAD_BASE_DIR, f"user_{current_user.id}", unique_name))

    try:
        logger.info(f"Cloning {git_url} to {proj_path}...")
        # Run blocking git call in thread pool
        await asyncio.to_thread(git.Repo.clone_from, git_url, proj_path, depth=1) # Shallow clone if full history not needed
        logger.info("Cloning complete.")
    except Exception as e:
        logger.error(f"Error cloning repo: {e}", exc_info=True)
        if os.path.exists(proj_path): import shutil; shutil.rmtree(proj_path)
        raise HTTPException(status_code=400, detail=f"Error cloning repo: {e}")

    # Validation
    if is_repo_empty(proj_path): raise HTTPException(status_code=400, detail="Repo empty.")
    if not has_code_files(proj_path): raise HTTPException(status_code=400, detail="No code files found.")

    # DB entries
    project = crud.create_project(db, user_id=current_user.id, project_name=project_name, file_name=git_url, local_path=proj_path)
    init = crud.create_initialization(db, user_id=current_user.id, project_id=project.id, persona=persona)

    # Trigger background analysis
    background_tasks.add_task(run_analysis_background, init.id, proj_path, [persona])
    logger.info(f"Scheduled git analysis: init_id={init.id}")

    return {"msg": f"Clone success. Analysis scheduled (Init ID: {init.id}).", "project_id": project.id, "initialization_id": init.id, "status": init.status}


# --- Endpoint to check analysis status/results (Essential for Background Task flow) ---
@router.get("/analysis/{initialization_id}", response_model=schemas.InitializationOut) # Changed path slightly
def get_analysis_status(initialization_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """ Gets the status and results of a specific analysis initialization record. """
    logger.info(f"Checking status for init_id {initialization_id}, user {current_user.id}")
    init = crud.get_initialization_by_id(db, initialization_id) # Use CRUD function

    if not init: raise HTTPException(status_code=404, detail="Initialization record not found.")
    if init.user_id != current_user.id: raise HTTPException(status_code=403, detail="Not authorized.")

    results_dict = None
    if init.status == "completed" and init.results:
        try: results_dict = json.loads(init.results)
        except Exception as e: results_dict = {"error": f"Failed to parse results: {e}"}

    # Map DB model to Pydantic schema
    # Ensure InitializationOut schema matches the fields being returned
    return schemas.InitializationOut(
        id=init.id, project_id=init.project_id, user_id=init.user_id,
        persona=init.persona, status=init.status, error_message=init.error_message,
        created_at=init.created_at, updated_at=init.updated_at,
        results=results_dict
    )
