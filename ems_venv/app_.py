@router.post("/upload/", response_model=dict)
async def upload_zip(
    project_name: str,
    persona: str,
    zip_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    logger.info(f"Zip upload: P='{project_name}', User={current_user.id}")
    if not ANALYSIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Analysis service unavailable.")
    if persona not in ("SDE", "PM"):
        raise HTTPException(status_code=400, detail="Invalid persona.")
    if not zip_file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Must be .zip")
    contents = await zip_file.read()
    if len(contents) > MAX_ZIP_SIZE:
        raise HTTPException(status_code=413, detail=f"File > {MAX_ZIP_SIZE//1024//1024}MB.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_name = f"{project_name.replace(' ','_')}_{timestamp}"
    proj_path = os.path.abspath(os.path.join(UPLOAD_BASE_DIR, f"user_{current_user.id}", unique_name))
    try:
        os.makedirs(proj_path, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(contents), 'r') as zf:
            zf.extractall(proj_path)
        logger.info(f"Extracted zip to: {proj_path}")
        if is_repo_empty(proj_path):
            raise ValueError("Extracted ZIP is empty.")
    except (zipfile.BadZipFile, ValueError) as e:
        if os.path.exists(proj_path): import shutil; shutil.rmtree(proj_path)
        raise HTTPException(status_code=400, detail=f"Invalid/Empty ZIP: {e}")
    except Exception as e:
        logger.error(f"Zip extract error: {e}", exc_info=True)
        if os.path.exists(proj_path): import shutil; shutil.rmtree(proj_path)
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    # DB entries
    project = crud.create_project(db, user_id=current_user.id, project_name=project_name, file_name=zip_file.filename, local_path=proj_path)
    init = crud.create_initialization(db, user_id=current_user.id, project_id=project.id, persona=persona)
    # --- START AGENT GRAPH FLOW (NO BG TASK) ---
    state = {
        "projectpath": proj_path,
        "personas": [persona],
        # You may need to add other required initial fields depending on your graph setup
    }
    try:
        analysis_graph = build_analysis_graph()    # Compile graph
        result_state = await analysis_graph.ainvoke(state) # Run graph and await response
        # Store results in DB (if your model requires it)
        init.status = "completed"
        init.results = json.dumps(result_state)
        db.commit()
    except Exception as e:
        logger.error(f"Agentic graph error: {e}", exc_info=True)
        init.status = "failed"
        init.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Agentic graph error: {e}")
    # --- RETURN FINAL STATE ---
    return {
        "msg": "Upload success. Analysis completed.",
        "project_id": project.id,
        "initialization_id": init.id,
        "status": init.status,
        "results": result_state,
    }

@router.post("/upload-git/", response_model=dict)
async def upload_git_repo(
    project_name: str,
    persona: str,
    git_url: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    logger.info(f"Git upload: P='{project_name}', URL='{git_url}', User={current_user.id}")
    if not ANALYSIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Analysis service unavailable.")
    if persona not in ("SDE", "PM"):
        raise HTTPException(status_code=400, detail="Invalid persona.")
    if not GIT_URL_REGEX.match(git_url):
        raise HTTPException(status_code=400, detail="Invalid git URL.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_name = f"{project_name.replace(' ','_')}_{timestamp}"
    proj_path = os.path.abspath(os.path.join(UPLOAD_BASE_DIR, f"user_{current_user.id}", unique_name))
    try:
        logger.info(f"Cloning {git_url} to {proj_path}...")
        await asyncio.to_thread(git.Repo.clone_from, git_url, proj_path, depth=1)
        logger.info("Cloning complete.")
        if is_repo_empty(proj_path):
            raise HTTPException(status_code=400, detail="Repo empty.")
        if not has_code_files(proj_path):
            raise HTTPException(status_code=400, detail="No code files found.")
    except Exception as e:
        logger.error(f"Error cloning repo: {e}", exc_info=True)
        if os.path.exists(proj_path): import shutil; shutil.rmtree(proj_path)
        raise HTTPException(status_code=400, detail=f"Error cloning repo: {e}")
    # DB entries
    project = crud.create_project(db, user_id=current_user.id, project_name=project_name, file_name=git_url, local_path=proj_path)
    init = crud.create_initialization(db, user_id=current_user.id, project_id=project.id, persona=persona)
    # --- AGENT GRAPH FLOW STARTS HERE ---
    state = {
        "projectpath": proj_path,
        "personas": [persona],
    }
    try:
        analysis_graph = build_analysis_graph()
        result_state = await analysis_graph.ainvoke(state)
        init.status = "completed"
        init.results = json.dumps(result_state)
        db.commit()
    except Exception as e:
        logger.error(f"Agentic graph error: {e}", exc_info=True)
        init.status = "failed"
        init.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Agentic graph error: {e}")
    return {
        "msg": "Clone success. Analysis completed.",
        "project_id": project.id,
        "initialization_id": init.id,
        "status": init.status,
        "results": result_state,
    }
