from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict
import uuid
from app import triage_system, session_states

# Ensure the app instance is defined at the module level
app = FastAPI()

# Mount static files (React build output)
app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the React frontend."""
    with open("frontend/build/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/api/triage", response_class=JSONResponse)
async def start_triage(payload: Dict[str, str] = Body(...)):
    """Start a new triage session."""
    problem = payload.get("problem")
    if not problem:
        return JSONResponse({"error": "Problem description is required."}, status_code=400)

    print(f"Received problem: {problem}")  # Debugging log
    session_id = str(uuid.uuid4())
    final_state_obj = triage_system.run_triage(problem, session_id)
    return {"session_id": session_id, "state": final_state_obj.get_state_summary()}

@app.post("/api/continue", response_class=JSONResponse)
async def continue_triage(payload: Dict[str, str] = Body(...)):
    """Continue an existing triage session."""
    session_id = payload.get("session_id")
    additional_input = payload.get("additional_input")

    if not session_id or not additional_input:
        return JSONResponse({"error": "Session ID and additional input are required."}, status_code=400)

    if session_id not in session_states:
        return JSONResponse({"error": "Session not found."}, status_code=404)

    updated_state = triage_system.continue_triage(session_id, additional_input)
    return {"session_id": session_id, "state": updated_state.get_state_summary()}
