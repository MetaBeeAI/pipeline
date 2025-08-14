# FastAPI backend webserver code
#
# m.mieskolainen@imperial.ac.uk, 2025

import os
import json
import fitz
from datetime import datetime
from typing import Any, Dict
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Dummy user credentials for demonstration.
# In production, replace this with a proper authentication system.
USERS = {
    "alice": "alicepassword",
    "bob":   "bobpassword",
}

BASE_PAPER_DIR = "data/papers"
FEEDBACK_DIR   = "data/feedback"  # Base directory for per-user feedback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Adjust for your real domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve PDFs and assets from /static/<paper_id> -> data/papers/<paper_id>
app.mount("/static", StaticFiles(directory=BASE_PAPER_DIR), name="static")

def log_action(paper: str, username: str, action: str) -> None:
    """
    Log the given action to the file feedback/<paper>/<username>/backend.log.
    The log entry includes a date-time stamp.
    """
    log_dir = os.path.join(FEEDBACK_DIR, paper, username)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "backend.log")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{now}] {action}\n")

@app.get("/login")
def login(
    username: str = Query(..., description="User ID"),
    password: str = Query(..., description="User password")
):
    """
    Login endpoint. Checks the provided credentials.
    If credentials are invalid, returns a 401 error.
    """
    if username not in USERS or USERS[username] != password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    # Optionally, log login action to a default file or omit logging here.
    return {"status": "ok"}

@app.get("/papers")
def list_papers():
    """
    Returns a sorted list of paper IDs (folder names containing digits only).
    """
    papers = [folder for folder in os.listdir(BASE_PAPER_DIR) if folder.isdigit()]
    papers.sort()
    return {"papers": papers}

@app.get("/json/{paper_id}")
def get_paper_json(paper_id: str):
    """
    Returns JSON containing data (chunks and questions) for the given paper.
    If the source JSON doesn't have a top-level 'data', it is wrapped accordingly.
    """
    path = os.path.join(BASE_PAPER_DIR, paper_id, "pages", "merged_v2.json")
    if not os.path.isfile(path):
        return JSONResponse(status_code=404, content={"error": "JSON not found"})
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "data" not in raw:
        raw = {"data": raw}
    return raw

def parse_questions(questions_data):
    """Parse the questions data structure to extract answers and reasons."""
    parsed = {}
    for q_key, q_val in questions_data.items():
        if isinstance(q_val, dict) and "answer" in q_val and "reason" in q_val:
            parsed[q_key] = q_val
        elif isinstance(q_val, dict):
            for sub_key, sub_val in q_val.items():
                if isinstance(sub_val, dict) and "answer" in sub_val and "reason" in sub_val:
                    new_key = f"{q_key}.{sub_key}"
                    parsed[new_key] = sub_val
    return parsed

@app.get("/answers/{paper_id}")
def get_answers(paper_id: str):
    """
    Returns the official AI answers with parsed questions.
    If no answers file is found, an empty QUESTIONS structure is returned.
    """
    answers_path = os.path.join(BASE_PAPER_DIR, paper_id, "answers.json")
    if not os.path.isfile(answers_path):
        return {"QUESTIONS": {}}
    with open(answers_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if "QUESTIONS" not in raw_data:
        raw_data = {"QUESTIONS": raw_data}
    raw_questions = raw_data["QUESTIONS"]
    parsed_questions = parse_questions(raw_questions)
    return {"QUESTIONS": parsed_questions}

@app.get("/answers_extended/{paper_id}")
def get_answers_extended(
    paper_id: str,
    username: str = Query(..., description="User ID"),
    password: str = Query(..., description="User password")
):
    """
    Returns the user's extended answers stored in a user-specific file.
    The file is expected at:
      data/feedback/{paper_id}/{username}/answers_extended.json
    If not found, returns an empty QUESTIONS structure and "Never" as modified_timestamp.
    Also logs the API action.
    """
    if username not in USERS or USERS[username] != password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Log the action.
    log_action(paper_id, username, f"GET /answers_extended/{paper_id}?username={username}")
    
    feedback_path = os.path.join(FEEDBACK_DIR, paper_id, username, "answers_extended.json")
    if not os.path.isfile(feedback_path):
        return {"QUESTIONS": {}, "modified_timestamp": "Never"}
    
    with open(feedback_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "QUESTIONS" not in data:
        data["QUESTIONS"] = {}
    mod_time = os.path.getmtime(feedback_path)
    modified_timestamp = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
    return {"QUESTIONS": data["QUESTIONS"], "modified_timestamp": modified_timestamp}


@app.post("/answers_extended/{paper_id}")
async def save_answers_extended(
    paper_id: str,
    request: Request,
    username: str = Query(..., description="User ID"),
    password: str = Query(..., description="User password")
):
    if username not in USERS or USERS[username] != password:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    log_action(paper_id, username, f"POST /answers_extended/{paper_id}?username={username}")

    user_feedback_dir = os.path.join(FEEDBACK_DIR, paper_id, username)
    os.makedirs(user_feedback_dir, exist_ok=True)
    feedback_path = os.path.join(user_feedback_dir, "answers_extended.json")

    try:
        body = await request.json()
        if "QUESTIONS" not in body:
            body["QUESTIONS"] = {}

        # Generate new timestamp
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body["modified_timestamp"] = now_str  # Save it inside the JSON

        # Save the file
        with open(feedback_path, "w", encoding="utf-8") as f:
            json.dump(body, f, indent=2)

        return {
            "status": "ok",
            "paper_id": paper_id,
            "modified_timestamp": now_str  # Also return it to frontend
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pdf/{paper_id}")
def get_pdf_metadata(paper_id: str):
    """
    Returns metadata for the PDF file (such as page count) for the given paper.
    """
    pdf_path = os.path.join(BASE_PAPER_DIR, paper_id, f"{paper_id}_main.pdf")
    if not os.path.isfile(pdf_path):
        return JSONResponse(status_code=404, content={"error": "PDF not found"})
    doc = fitz.open(pdf_path)
    return {"pages": len(doc)}
