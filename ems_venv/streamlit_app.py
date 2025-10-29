
# streamlit_multiagent_ui.py
import json
import threading
from typing import Optional, Dict, Any, List

import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    import websocket
    HAS_WSCLIENT = True
except Exception:
    HAS_WSCLIENT = False

st.set_page_config(page_title="Multi-Agent Docs — Admin & User", layout="wide")

# -------------------- Helpers --------------------

def api_headers(token: Optional[str]) -> Dict[str, str]:
    h = {"Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def api_call(method: str, url: str, token: Optional[str] = None, **kwargs) -> requests.Response:
    try:
        resp = requests.request(method, url, headers=api_headers(token), **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            st.error(f"API error {resp.status_code}: {detail}")
        return resp
    except Exception as e:
        st.error(f"Request failed: {e}")
        raise

def render_mermaid(mermaid_code: str, height: int = 360):
    html = f"""
    <div id="mmd" class="mermaid">
    {mermaid_code}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{ startOnLoad: true, securityLevel: 'loose' }});</script>
    """
    components.html(html, height=height, scrolling=True)

# -------------------- Session State --------------------

SS = st.session_state
SS.setdefault("token", None)
SS.setdefault("me", None)
SS.setdefault("role", "user")
SS.setdefault("project_id", None)
SS.setdefault("ws_messages", [])
SS.setdefault("show_signup", False)
SS.setdefault("show_config_for", None)
SS.setdefault("show_analyze_for", None)

# -------------------- Sidebar Config --------------------

st.sidebar.header("Backend Settings")
API_BASE = st.sidebar.text_input("API Base URL", value="http://localhost:8000").rstrip("/")
WS_BASE = st.sidebar.text_input("WS Base", value="ws://localhost:8000").rstrip("/")
WS_PATH = st.sidebar.text_input("WS Path (use {project_id})", value="/analysis/ws/{project_id}")

# -------------------- Auth Panel --------------------

st.title("Code Analyzer")

auth_col1, auth_col2 = st.columns([3,2])
with auth_col1:
    if not SS["token"] or SS["show_signup"]:
        st.subheader("Sign up" if SS["show_signup"] else "Log in")
        if SS["show_signup"]:
            su_name = st.text_input("Name")
            su_email = st.text_input("Email")
            su_pw = st.text_input("Password", type="password")
            if st.button("Create account"):
                resp = api_call("POST", f"{API_BASE}/users/", json={"name": su_name, "email": su_email, "password": su_pw})
                if resp.ok:
                    st.success("Account created. Please log in.")
                    SS["show_signup"] = False
                    st.rerun()
        else:
            li_email = st.text_input("Email")
            li_pw = st.text_input("Password", type="password")
            c1, c2 = st.columns([1,1])
            if c1.button("Log in"):
                resp = api_call("POST", f"{API_BASE}/token", json={"email": li_email, "password": li_pw})
                if resp.ok:
                    tok = resp.json().get("access_token")
                    SS["token"] = tok
                    me = api_call("GET", f"{API_BASE}/me", token=SS["token"])
                    if me.ok:
                        SS["me"] = me.json()
                        SS["role"] = SS["me"].get("role","user").lower()
                    st.rerun()
            if c2.button("Sign up"):
                SS["show_signup"] = True
                st.rerun()
    else:
        st.markdown(f"**Logged in as:** {SS['me'].get('email','')}  \n**Role:** {SS['role'].title()}")
        if st.button("Log out"):
            SS["token"] = None
            SS["me"] = None
            SS["project_id"] = None
            st.rerun()

# -------------------- Main Views --------------------

if SS["token"]:
    tabs = ["Projects"]
    if SS["role"] == "admin":
        tabs += ["Admin: Users", "Admin: Projects", "Admin: Analytics"]
    sel = st.tabs(tabs)

    # -------- Regular User: Projects --------
    with sel[0]:
        st.subheader("Your Projects")
        top = st.container()
        with top:
            c1, c2 = st.columns([6,1])
            c1.write("List of projects associated with your account.")
            if c2.button("+ Create"):
                SS["show_config_for"] = {"id": None, "name": "", "personas": ["SDE","PM"], "enable_web_search": False, "depth":"light", "verbosity":"standard"}

        # Fetch projects (needs backend /projects)
        resp = api_call("GET", f"{API_BASE}/projects", token=SS["token"])
        data = resp.json() if resp.ok else []
        for p in data or []:
            with st.container(border=True):
                left, right = st.columns([7,3])
                with left:
                    st.markdown(f"**#{p.get('id')} — {p.get('name')}**")
                    st.caption(f"Status: {p.get('status','unknown')} • Personas: {p.get('personas','')}")
                with right:
                    colA, colB, colC = st.columns(3)
                    if colA.button("View / Edit", key=f"edit_{p['id']}"):
                        # open config popup
                        SS["show_config_for"] = {
                            "id": p["id"],
                            "name": p.get("name",""),
                            "personas": p.get("personas","SDE,PM").split(","),
                            "enable_web_search": False,
                            "depth":"light",
                            "verbosity":"standard",
                        }
                    if colB.button("Analyze", key=f"analyze_{p['id']}"):
                        SS["show_analyze_for"] = {"id": p["id"]}
                    if colC.button("Delete", key=f"del_{p['id']}"):
                        api_call("DELETE", f"{API_BASE}/projects/{p['id']}", token=SS["token"])
                        st.experimental_rerun()

        # Configuration popup
        cfg = SS.get("show_config_for")
        if cfg is not None:
            with st.expander("Project Configuration", expanded=True):
                name = st.text_input("Project Name", value=cfg.get("name",""))
                personas = st.multiselect("Personas", ["SDE","PM"], default=cfg.get("personas",["SDE","PM"]))
                enable_web = st.toggle("Enable Web Search", value=cfg.get("enable_web_search",False))
                depth = st.selectbox("Analysis Depth", ["light","mid","deep"], index=["light","mid","deep"].index(cfg.get("depth","light")))
                verbosity = st.selectbox("Verbosity", ["low","standard","high"], index=["low","standard","high"].index(cfg.get("verbosity","standard")))

                c1, c2, c3 = st.columns(3)
                if c1.button("Save"):
                    payload = {
                        "name": name,
                        "personas": ",".join(personas),
                    }
                    if cfg.get("id"):
                        api_call("PATCH", f"{API_BASE}/projects/{cfg['id']}", token=SS["token"], json=payload)
                    else:
                        api_call("POST", f"{API_BASE}/projects", token=SS["token"], json=payload)
                    SS["show_config_for"] = None
                    st.experimental_rerun()
                if c2.button("Start Analysis"):
                    payload = {
                        "project_id": str(cfg.get("id","")),
                        "personas": personas,
                        "enable_web_search": enable_web,
                        "analysis_depth": depth,
                        "verbosity": verbosity,
                    }
                    api_call("POST", f"{API_BASE}/start-analysis/", token=SS["token"], json=payload)
                    st.success("Analysis started (config sent).")
                if c3.button("Close"):
                    SS["show_config_for"] = None
                    st.experimental_rerun()

        # Analyze (prompt) popup
        ap = SS.get("show_analyze_for")
        if ap is not None:
            with st.expander("Ask a Question (RAG)", expanded=True):
                q = st.text_input("Prompt", placeholder="e.g., How does authentication work here?")
                if st.button("Ask"):
                    payload = {"query": q, "final_state": {}}
                    r = api_call("POST", f"{API_BASE}/search", token=SS["token"], json=payload)
                    if r.ok:
                        ans = r.json().get("answer","")
                        st.markdown(ans)
                if st.button("Close", key="close_analyze"):
                    SS["show_analyze_for"] = None
                    st.experimental_rerun()

    # -------- Admin: Users --------
    if SS["role"] == "admin":
        with sel[1]:
            st.subheader("All Users")
            r = api_call("GET", f"{API_BASE}/admin/users", token=SS["token"])
            rows = r.json() if r.ok else []
            st.dataframe(rows, use_container_width=True)

        # -------- Admin: Projects --------
        with sel[2]:
            st.subheader("All Projects")
            r = api_call("GET", f"{API_BASE}/projects", token=SS["token"])
            rows = r.json() if r.ok else []
            st.dataframe(rows, use_container_width=True)
            st.caption("Use per-row actions in the user view to edit/delete individual projects.")

        # -------- Admin: Analytics --------
        with sel[3]:
            st.subheader("Analytics")
            r = api_call("GET", f"{API_BASE}/admin/analytics", token=SS["token"])
            if r.ok:
                stats = r.json()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total users", stats.get("total_users", 0))
                c2.metric("Total projects", stats.get("total_projects", 0))
                c3.metric("Active analyses", stats.get("active_analyses", 0))
            else:
                st.info("Add an /admin/analytics endpoint on the backend to power this view.")

# -------------------- Footer --------------------

st.caption("UI wired for your routes.py with extra conventional endpoints for projects. Add them on the backend if missing.")