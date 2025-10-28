# --- api/routes.py (Corrected BackgroundTasks Dependency) ---

import hashlib
import io
import os
import time
from datetime import datetime, timedelta
import re
import zipfile
import json
import logging
import asyncio

# FastAPI imports
from fastapi import (
    APIRouter, Depends, File, HTTPException, UploadFile, status,
    BackgroundTasks # Ensure BackgroundTasks is imported
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Other imports
import git
from sqlalchemy.orm import Session
from jose import JWTError, jwt

# --- Your project imports (ADJUST PATHS if needed) ---
try:
    from multi_agent_docs.graph.graph import build_analysis_graph, CodeAnalysisState
    ANALYSIS_AVAILABLE = True
    logging.info("Successfully imported graph components.")
except ImportError as e:
    ANALYSIS_AVAILABLE = False
    build_analysis_graph = None
    CodeAnalysisState = dict
    print(f"\n--- WARNING --- \nCould not import graph: {e}\nAnalysis trigger disabled.\n")
except Exception as e:
    ANALYSIS_AVAILABLE = False
    build_analysis_graph = None
    CodeAnalysisState = dict
    print(f"\n--- WARNING --- \nUnexpected error importing graph: {e}\nAnalysis trigger disabled.\n")

# Database/Auth related imports
from models.model import Initialisation, User, Project
from crud import crud
from database.db import get_db, SessionLocal
from schema import schemas

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()
UPLOAD_BASE_DIR = os.path.abspath("uploads")
os.makedirs(UPLOAD_BASE_DIR, exist_ok=True)

# --- JWT, Security, Regex, Constants (Keep as before) ---
SECRET_KEY = os.getenv("SECRET_KEY", "SECRET!ChangeMe!")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
MAX_ZIP_SIZE = 100 * 1024 * 1024
GIT_URL_REGEX = re.compile(r"^(https://|git@|ssh://)[\w\.\-/:]+\.git$")
ALLOWED_EXTENSIONS = {'.py', '.js', '.java', '.c', '.cpp', '.ts', '.go', '.rb', '.php', '.cs', '.rs'}

security = HTTPBearer()

# --- Helper Functions (Keep as before) ---
def create_access_token(data: dict, expires_delta: timedelta = None):
    # ...(implementation)...
    to_encode=data.copy(); expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire}); return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    # ...(implementation)...
    token = credentials.credentials; cred_exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})
    try: payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]); email: str = payload.get("sub");
        if email is None: raise cred_exc
    except JWTError: raise cred_exc
    user = crud.get_user_by_email(db, email);
    if user is None: raise cred_exc
    return user

def has_code_files(repo_path):
    # ...(implementation)...
    if not os.path.isdir(repo_path): return False
    for root, _, files in os.walk(repo_path):
        for file in files: _, ext = os.path.splitext(file);
            if ext.lower() in ALLOWED_EXTENSIONS: return True
    return False

def is_repo_empty(repo_path):
    # ...(implementation)...
     if not os.path.isdir(repo_path): return True
     for _, _, files in os.walk(repo_path):
        if files: return False
     return True

# --- User Endpoints (/users/, /token, /me - Keep as before) ---
@router.post("/users/", response_model=schemas.UserOut, tags=["Authentication"])
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # ...(implementation)...
    if crud.get_user_by_email(db, user.email): raise HTTPException(status_code=400, detail="Email registered")
    hashed_pw = hashlib.sha256(user.password.encode()).hexdigest()
    new_user = crud.create_user(db, name=user.name, email=user.email, hashed_password=hashed_pw)
    return new_user

@router.post("/token", tags=["Authentication"])
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    # ...(implementation)...
    db_user = crud.get_user_by_email(db, user.email); hashed_pw = hashlib.sha256(user.password.encode()).hexdigest()
    if not db_user or db_user.password != hashed_pw: raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(data={"sub": db_user.email}); return {"access_token": token, "token_type": "bearer"}

@router.get("/me", response_model=schemas.UserOut, tags=["Users"])
def read_current_user(current_user: User = Depends(get_current_user)):
    # ...(implementation)...
    return current_user

# --- Background Task Function (Keep as before) ---
async def run_analysis_background(initialization_id: int, project_path: str, personas: List[str]):
    """ Compiles/runs LangGraph analysis, updates DB status/results. """
    db: Session = SessionLocal()
    analysis_graph = None
    graph_compiled = False
    try:
        logger.info(f"[BG Task {initialization_id}] Starting analysis: {project_path}")
        if build_analysis_graph is None: raise ImportError("Graph build function unavailable.")
        crud.update_initialization_status(db, initialization_id, "processing")
        try:
             logger.info(f"[BG Task {initialization_id}] Compiling graph...")
             analysis_graph = build_analysis_graph() # Compile sync
             if not analysis_graph: raise RuntimeError("Graph compilation returned None.")
             graph_compiled = True; logger.info(f"[BG Task {initialization_id}] Graph compiled.")
        except Exception as compile_err:
             logger.error(f"[BG Task {initialization_id}] Compile failed: {compile_err}", exc_info=True)
             crud.update_initialization_status(db, initialization_id, "failed", f"Compile error: {compile_err}")
             return
        initial_state = CodeAnalysisState(project_path=project_path, personas=personas)
        config = {"recursion_limit": 15}
        logger.info(f"[BG Task {initialization_id}] Invoking graph...")
        timeout_seconds = 1800
        try: final_state = await asyncio.wait_for(analysis_graph.ainvoke(initial_state, config=config), timeout=timeout_seconds)
        except asyncio.TimeoutError: logger.error(f"[BG Task {initialization_id}] Timeout after {timeout_seconds}s."); crud.update_initialization_status(db, initialization_id, "failed", f"Timeout after {timeout_seconds}s."); return
        logger.info(f"[BG Task {initialization_id}] Graph finished.")
        final_status = "completed"; error_msg = final_state.get("code_discovery_error") or final_state.get("file_proc_error") or final_state.get("repo_intel_error")
        if error_msg:
             final_status = "failed"; logger.error(f"[BG Task {initialization_id}] Analysis failed: {error_msg}"); crud.update_initialization_status(db, initialization_id, final_status, str(error_msg))
        else:
             try:
                 keys_to_store = ["tech_stack", "structure_summary", "architecture_summary", "complexity_assessment", "code_elements", "dependencies", "file_relationships", "suggested_skip_patterns", "semantic_chunks", "personas", "repo_files"]
                 result_to_store = {k: final_state.get(k) for k in keys_to_store if final_state.get(k) is not None}
                 result_json = json.dumps(result_to_store, default=str)
                 crud.store_analysis_result(db, initialization_id, result_json)
                 crud.update_initialization_status(db, initialization_id, final_status)
                 logger.info(f"[BG Task {initialization_id}] Analysis completed & results stored.")
             except Exception as store_err: logger.error(f"[BG Task {initialization_id}] Store results failed: {store_err}", exc_info=True); crud.update_initialization_status(db, initialization_id, "failed", f"Result storage error: {store_err}")
    except ImportError as imp_err: logger.error(f"[BG Task {initialization_id}] Import error: {imp_err}", exc_info=True); crud.update_initialization_status(db, initialization_id, "failed", f"Import Error: {imp_err}")
    except Exception as e:
        logger.error(f"[BG Task {initialization_id}] UNEXPECTED ERROR: {e}", exc_info=True)
        try: init = crud.get_initialization_by_id(db, initialization_id);
            if init and init.status != "failed": crud.update_initialization_status(db, initialization_id, "failed", f"Unexpected background error: {e}")
        except Exception as db_err: logger.error(f"[BG Task {initialization_id}] FAILED to update status after error: {db_err}")
    finally: db.close()


# --- Upload Endpoints (Corrected BackgroundTasks Dependency) ---
@router.post("/upload/", response_model=dict, tags=["Project Initiation"])
async def upload_zip(
    project_name: str,
    persona: str,
    zip_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    # --- CORRECTED: Remove Depends() ---
    background_tasks: BackgroundTasks
):
    logger.info(f"Zip upload: P='{project_name}', User={current_user.id}, Persona={persona}")
    if not ANALYSIS_AVAILABLE: raise HTTPException(status_code=503, detail="Analysis service unavailable.")
    if persona not in ("SDE", "PM"): raise HTTPException(status_code=400, detail="Invalid persona ('SDE' or 'PM').")
    if not zip_file.filename.lower().endswith(".zip"): raise HTTPException(status_code=400, detail="Must be .zip")

    contents = await zip_file.read()
    if len(contents) > MAX_ZIP_SIZE: raise HTTPException(status_code=413, detail=f"File > {MAX_ZIP_SIZE//1024//1024}MB.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_project_name = re.sub(r'[^\w\-]+', '_', project_name)
    unique_name = f"{safe_project_name}_{timestamp}"
    proj_path = os.path.abspath(os.path.join(UPLOAD_BASE_DIR, f"user_{current_user.id}", unique_name))

    try:
        os.makedirs(proj_path, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(contents), 'r') as zf: zf.extractall(proj_path)
        logger.info(f"Extracted zip to: {proj_path}")
        if is_repo_empty(proj_path): raise ValueError("Extracted ZIP is empty.")
        if not has_code_files(proj_path): raise ValueError("No recognized code files found.")
    except (zipfile.BadZipFile, ValueError) as e:
        if os.path.exists(proj_path): import shutil; shutil.rmtree(proj_path)
        raise HTTPException(status_code=400, detail=f"Invalid/Empty/Non-code ZIP: {e}")
    except Exception as e:
         logger.error(f"Zip extract error: {e}", exc_info=True)
         if os.path.exists(proj_path): import shutil; shutil.rmtree(proj_path)
         raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    project = crud.create_project(db, user_id=current_user.id, project_name=project_name, file_name=zip_file.filename, local_path=proj_path)
    init = crud.create_initialization(db, user_id=current_user.id, project_id=project.id, persona=persona)

    background_tasks.add_task(run_analysis_background, init.id, proj_path, [persona])
    logger.info(f"Scheduled zip analysis: init_id={init.id}")

    return {"msg": f"Upload success. Analysis scheduled (Init ID: {init.id}).", "project_id": project.id, "initialization_id": init.id, "status": init.status}

@router.post("/upload-git/", response_model=dict, tags=["Project Initiation"])
async def upload_git_repo(
    project_name: str,
    persona: str,
    git_url: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    # --- CORRECTED: Remove Depends() ---
    background_tasks: BackgroundTasks
):
    logger.info(f"Git upload: P='{project_name}', URL='{git_url}', User={current_user.id}, Persona={persona}")
    if not ANALYSIS_AVAILABLE: raise HTTPException(status_code=503, detail="Analysis service unavailable.")
    if persona not in ("SDE", "PM"): raise HTTPException(status_code=400, detail="Invalid persona.")
    if not GIT_URL_REGEX.match(git_url): raise HTTPException(status_code=400, detail="Invalid git URL.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_project_name = re.sub(r'[^\w\-]+', '_', project_name)
    unique_name = f"{safe_project_name}_{timestamp}"
    proj_path = os.path.abspath(os.path.join(UPLOAD_BASE_DIR, f"user_{current_user.id}", unique_name))

    try:
        logger.info(f"Cloning {git_url} to {proj_path}...")
        await asyncio.to_thread(git.Repo.clone_from, git_url, proj_path, depth=1)
        logger.info("Cloning complete.")
    except Exception as e:
        logger.error(f"Error cloning repo: {e}", exc_info=True)
        if os.path.exists(proj_path): import shutil; shutil.rmtree(proj_path)
        raise HTTPException(status_code=400, detail=f"Error cloning repo: {e}")

    if is_repo_empty(proj_path): raise HTTPException(status_code=400, detail="Repo empty.")
    if not has_code_files(proj_path): raise HTTPException(status_code=400, detail="No code files found.")

    project = crud.create_project(db, user_id=current_user.id, project_name=project_name, file_name=git_url, local_path=proj_path)
    init = crud.create_initialization(db, user_id=current_user.id, project_id=project.id, persona=persona)

    background_tasks.add_task(run_analysis_background, init.id, proj_path, [persona])
    logger.info(f"Scheduled git analysis: init_id={init.id}")

    return {"msg": f"Clone success. Analysis scheduled (Init ID: {init.id}).", "project_id": project.id, "initialization_id": init.id, "status": init.status}

# --- Status Endpoint Removed ---
