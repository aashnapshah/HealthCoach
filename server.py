"""
HealthCoach Server
Handles the web interface and chat flow
"""

import os
from typing import Dict
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix

# App and static config
STATIC_DIR = "/n/data1/hms/dbmi/manrai/aashna/HealthCoachV2/static"
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
app.wsgi_app = ProxyFix(app.wsgi_app)

# Import domain modules
from ehr_data import PATIENT_DATA, get_patient_data, get_patient_names
from chat_manager import ChatManager

# Simple in-memory sessions keyed by patient_id (demo)
CHAT_SESSIONS: Dict[str, ChatManager] = {}

def to_ui_payload(q: dict) -> dict:
    """Map question format to UI payload"""
    if not q:
        return {"complete": True, "summary": "No questions available."}
    
    # Handle transition messages
    if q.get("type") in ["transition", "complete"]:
        return {
            "type": "section_header",
            "question": q["question"],
            "section": q["section"]
        }
    
    # Handle regular questions
    qt = q.get("type", "text")
    if qt == "yes_no_maybe":
        qt = "yes_no"
    
    # Get section from question
    section = q.get("section", "verification")
    
    return {
        "question": q.get("question", ""),
        "question_type": qt,
        "options": q.get("options"),
        "scale_labels": q.get("scale_labels"),
        "question_key": q.get("key"),
        "section": section
    }

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/patient_selection.html")
def patient_selection():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/intake.html")
def intake_page():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/visit_prep.html")
def visit_prep_page():
    return send_from_directory(STATIC_DIR, "visit_prep.html")

@app.route("/start_visit_prep", methods=["POST"])
def start_visit_prep():
    try:
        print("DEBUG: Starting visit prep...")
        data = request.get_json(force=True) or {}
        patient_id = str(data.get("patient_id", "")).upper()
        print(f"DEBUG: Patient ID: {patient_id}")
        
        patient = get_patient_data(patient_id)
        if not patient:
            print(f"ERROR: Invalid patient_id: {patient_id}")
            return jsonify({"error": "Invalid patient_id"}), 400

        print(f"DEBUG: Found patient: {patient.get('name')}")

        # Create a new chat manager specifically for visit prep
        manager = ChatManager(patient_id, patient, start_at_visit_prep=True)
        CHAT_SESSIONS[patient_id] = manager
        
        # Get first question
        print("DEBUG: Getting first question...")
        question, summary = manager.get_next_question()
        print(f"DEBUG: Question received: {question}")
        
        response = to_ui_payload(question)
        print(f"DEBUG: Response payload: {response}")
        
        # Add summary if provided
        if summary:
            response["summary"] = summary
            
        return jsonify(response)
    except Exception as e:
        print(f"ERROR in start_visit_prep: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route("/patients", methods=["GET"])
def patients():
    out = []
    for pid, pdata in PATIENT_DATA.items():
        out.append({
            "id": pid,
            "name": pdata.get("name"),
            "age": pdata.get("age"),
            "gender": pdata.get("gender"),
            "dob": pdata.get("dob"),
        })
    return jsonify(out)

@app.route("/start_intake", methods=["POST"])
def start_intake():
    try:
        print("DEBUG: Starting intake...")
        data = request.get_json(force=True) or {}
        patient_id = str(data.get("patient_id", "")).upper()
        print(f"DEBUG: Patient ID: {patient_id}")
        
        patient = get_patient_data(patient_id)
        if not patient:
            print(f"ERROR: Invalid patient_id: {patient_id}")
            return jsonify({"error": "Invalid patient_id"}), 400

        print(f"DEBUG: Found patient: {patient.get('name')}")

        # Create/restart chat manager for this patient
        manager = ChatManager(patient_id, patient)
        CHAT_SESSIONS[patient_id] = manager
        
        # Get first question
        print("DEBUG: Getting first question...")
        question, summary = manager.get_next_question()
        print(f"DEBUG: Question received: {question}")
        
        response = to_ui_payload(question)
        print(f"DEBUG: Response payload: {response}")
        
        # Add summary if provided
        if summary:
            response["summary"] = summary
            
        return jsonify(response)
    except Exception as e:
        print(f"ERROR in start_intake: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    data = request.get_json(force=True) or {}
    patient_id = str(data.get("patient_id", "")).upper()
    answer = data.get("answer")
    qkey = data.get("question_key")

    manager = CHAT_SESSIONS.get(patient_id)
    if not manager:
        return jsonify({"error": "Session not found. Please start again."}), 400

    # Submit answer and get next question
    question, summary = manager.submit_answer(answer, qkey)
    
    # Handle completion
    if manager.stage.name.lower() == "complete":
        return jsonify({
            "complete": True,
            "summary": summary
        })
    
    # Handle regular questions and transitions
    response = to_ui_payload(question)
    if summary:
        response["summary"] = summary
        
    return jsonify(response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)