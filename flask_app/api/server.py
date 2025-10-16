"""
HealthCoach Server
Handles the web interface and chat flow
"""

import sys
import os
import time

print("\n" + "="*70)
print("⏱️  [SERVER] Server module loading started...")
_server_load_start = time.time()

import requests
from typing import Dict
from flask import Flask, jsonify, request, render_template
from werkzeug.middleware.proxy_fix import ProxyFix

print(f"⏱️  [SERVER] Flask imports done ({time.time() - _server_load_start:.2f}s)")

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
flask_app_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(flask_app_dir)

# Add flask_app to path for config
sys.path.insert(0, flask_app_dir)
print(f"⏱️  [SERVER] Loading Config...")
_config_start = time.time()
from config import Config
print(f"⏱️  [SERVER] Config loaded ({time.time() - _config_start:.2f}s)")

# App setup (paths relative to this file)
print(f"⏱️  [SERVER] Creating Flask app...")
_app_start = time.time()
app = Flask(__name__, 
            static_folder=os.path.join(flask_app_dir, 'static'),
            static_url_path='/static',
            template_folder=os.path.join(flask_app_dir, 'templates'))
app.config.from_object(Config)
app.wsgi_app = ProxyFix(app.wsgi_app)
print(f"⏱️  [SERVER] Flask app created ({time.time() - _app_start:.2f}s)")

# Import domain modules
print(f"⏱️  [SERVER] Loading domain modules (ehr_data, chat_manager)...")
_domain_start = time.time()
sys.path.insert(0, os.path.join(project_root, 'data'))
from ehr_data import PATIENT_DATA, get_patient_data, get_patient_names
print(f"⏱️  [SERVER] ehr_data loaded ({time.time() - _domain_start:.2f}s)")

_chat_start = time.time()
from chat_manager import ChatManager
print(f"⏱️  [SERVER] chat_manager loaded ({time.time() - _chat_start:.2f}s)")
print(f"✓ [SERVER] Server module loaded ({time.time() - _server_load_start:.2f}s total)")
print("="*70 + "\n")

def check_model_availability():
    """Check if Hugging Face models are accessible"""
    try:
        from transformers import AutoTokenizer
        # Try to load the tokenizer as a simple check
        tokenizer = AutoTokenizer.from_pretrained(
            Config.HF_MODEL,
            token=Config.HF_TOKEN,
            cache_dir=Config.HF_CACHE_DIR
        )
        print(f"✓ Successfully verified access to Hugging Face model: {Config.HF_MODEL}")
        return True
    except Exception as e:
        print(f"❌ Failed to access Hugging Face model: {e}")
        print(f"  • Model: {Config.HF_MODEL}")
        print("  • This model will be downloaded on first use")
        print("  • Set HF_TOKEN environment variable if using a gated model (like Llama)")
        return False

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

@app.route("/health")
def health():
    """Simple health check endpoint"""
    print("\n💚 API: /health endpoint called")
    return jsonify({"status": "ok", "message": "Server is running!"})

@app.route("/")
def index():
    import time
    _start = time.time()
    print("\n🌐 [SERVER] Route '/' accessed - loading patient selection page")
    result = render_template("patient_selection.html")
    print(f"   ✓ Template rendered ({time.time() - _start:.3f}s)")
    return result

@app.route("/patient_selection.html")
def patient_selection():
    print("\n🌐 DEBUG: Route '/patient_selection.html' accessed")
    return render_template("patient_selection.html")

@app.route("/intake.html")
def intake_page():
    print("\n🌐 DEBUG: Route '/intake.html' accessed")
    return render_template("intake_continuous.html")

@app.route("/visit_prep.html")
def visit_prep_page():
    print("\n🌐 DEBUG: Route '/visit_prep.html' accessed")
    return render_template("visit_prep_chat.html")

@app.route("/visit_prep_chat.html")
def visit_prep_chat_page():
    print("\n🌐 DEBUG: Route '/visit_prep_chat.html' accessed")
    return render_template("visit_prep_chat.html")

@app.route("/intake_continuous.html")
def intake_continuous_page():
    print("\n🌐 DEBUG: Route '/intake_continuous.html' accessed")
    return render_template("intake_continuous.html")

@app.route("/visit_prep_continuous.html")
def visit_prep_continuous_page():
    print("\n🌐 DEBUG: Route '/visit_prep_continuous.html' accessed")
    return render_template("visit_prep_continuous.html")

@app.route("/final_summary.html")
def final_summary_page():
    print("\n🌐 DEBUG: Route '/final_summary.html' accessed")
    return render_template("final_summary.html")

@app.route("/start_visit_prep", methods=["POST"])
def start_visit_prep():
    try:
        print("\n" + "="*60)
        print("📋 API: start_visit_prep endpoint called")
        print("="*60)
        data = request.get_json(force=True) or {}
        patient_id = str(data.get("patient_id", "")).upper()
        print(f"  ➜ Patient ID: {patient_id}")
        
        patient = get_patient_data(patient_id)
        if not patient:
            print(f"  ❌ ERROR: Invalid patient_id: {patient_id}")
            return jsonify({"error": "Invalid patient_id"}), 400

        print(f"  ✓ Found patient: {patient.get('name')}")

        # Create a new chat manager specifically for visit prep
        print(f"  ➜ Creating ChatManager...")
        manager = ChatManager(patient_id, patient, start_at_visit_prep=True)
        CHAT_SESSIONS[patient_id] = manager
        print(f"  ✓ ChatManager created")
        
        # Get first question
        print(f"  ➜ Getting first question (LLM will load here if needed)...")
        question, summary = manager.get_next_question()
        print(f"  ✓ Question received: {type(question)}")
        
        response = to_ui_payload(question)
        print(f"  ✓ Response prepared, sending to client")
        print("="*60 + "\n")
        
        # Add summary if provided
        if summary:
            response["summary"] = summary
            
        return jsonify(response)
    except Exception as e:
        print(f"\n❌ EXCEPTION in start_visit_prep: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route("/patients", methods=["GET"])
def patients():
    import time
    _start = time.time()
    print("\n📋 [SERVER] /patients endpoint called")
    print(f"   • PATIENT_DATA type: {type(PATIENT_DATA)}")
    print(f"   • PATIENT_DATA keys: {list(PATIENT_DATA.keys())}")
    print(f"   • Number of patients: {len(PATIENT_DATA)}")
    
    out = []
    for pid, pdata in PATIENT_DATA.items():
        patient_info = {
            "id": pid,
            "name": pdata.get("name"),
            "age": pdata.get("age"),
            "gender": pdata.get("gender"),
            "dob": pdata.get("dob"),
        }
        print(f"   • Processing patient {pid}: {patient_info}")
        out.append(patient_info)
    
    print(f"   ✓ Returning {len(out)} patients ({time.time() - _start:.3f}s)")
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

@app.route("/get_all_intake_questions", methods=["POST"])
def get_all_intake_questions():
    """Get all intake questions at once for continuous form"""
    try:
        print("DEBUG: get_all_intake_questions called")
        data = request.get_json(force=True) or {}
        patient_id = str(data.get("patient_id", "")).upper()
        print(f"DEBUG: patient_id = {patient_id}")
        
        if not patient_id:
            return jsonify({"error": "Missing patient_id"}), 400
        
        patient = get_patient_data(patient_id)
        if not patient:
            return jsonify({"error": "Invalid patient_id"}), 400
        
        print(f"DEBUG: Found patient: {patient.get('name')}")
        
        # Create a new chat manager for intake
        print("DEBUG: Creating ChatManager...")
        manager = ChatManager(patient_id, patient)
        CHAT_SESSIONS[patient_id] = manager
        print(f"DEBUG: ChatManager created, intake_session exists: {manager.intake_session is not None}")
        
        # Get all intake questions directly from the session
        print("DEBUG: Getting verification questions...")
        all_questions = manager.intake_session.get_verification_questions()
        print(f"DEBUG: Got {len(all_questions)} questions")
        
        # Convert to UI format
        questions = []
        for i, q in enumerate(all_questions):
            print(f"DEBUG: Processing question {i+1}: {q.get('question', '')[:50]}...")
            questions.append({
                "question": q.get("question", ""),
                "question_type": q.get("type", "text"),
                "options": q.get("options"),
                "scale_labels": q.get("scale_labels"),
                "question_key": q.get("key"),
                "context": q.get("context"),
                "section": "verification",
                "multiselect": q.get("allow_multiple", False)
            })
        
        print(f"DEBUG: Returning {len(questions)} questions to frontend")
        return jsonify({"questions": questions})
    except Exception as e:
        print(f"ERROR in get_all_intake_questions: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/get_all_visit_prep_questions", methods=["POST"])
def get_all_visit_prep_questions():
    """Get all visit prep questions at once for continuous form"""
    try:
        print("DEBUG: get_all_visit_prep_questions called")
        data = request.get_json(force=True) or {}
        patient_id = str(data.get("patient_id", "")).upper()
        print(f"DEBUG: patient_id = {patient_id}")
        
        if not patient_id:
            return jsonify({"error": "Missing patient_id"}), 400
        
        patient = get_patient_data(patient_id)
        if not patient:
            return jsonify({"error": "Invalid patient_id"}), 400
        
        print(f"DEBUG: Found patient: {patient.get('name')}")
        
        # Get or create chat manager for visit prep
        manager = CHAT_SESSIONS.get(patient_id)
        if not manager:
            print("DEBUG: Creating new ChatManager for visit prep")
            manager = ChatManager(patient_id, patient, start_at_visit_prep=True)
            CHAT_SESSIONS[patient_id] = manager
        else:
            print("DEBUG: Using existing ChatManager")
        
        # Initialize visit_session if it doesn't exist
        if not manager.visit_session:
            print("DEBUG: Initializing visit_session")
            from visit_prep import VisitSession
            intake_answers = getattr(manager.intake_session, 'responses', {}) if manager.intake_session else {}
            manager.visit_session = VisitSession(patient_id, patient, intake_answers)
        
        # Get all visit prep questions directly from the session
        print("DEBUG: Getting visit questions...")
        all_questions = manager.visit_session.get_visit_questions()
        print(f"DEBUG: Got {len(all_questions)} questions")
        
        # Convert to UI format
        questions = []
        for q in all_questions:
            questions.append({
                "question": q.get("question", ""),
                "question_type": q.get("type", "text"),
                "options": q.get("options"),
                "scale_labels": q.get("scale_labels"),
                "question_key": q.get("key"),
                "context": q.get("context"),
                "section": "visit_prep",
                "multiselect": q.get("allow_multiple", False)
            })
        
        print(f"DEBUG: Returning {len(questions)} questions to frontend")
        return jsonify({"questions": questions})
    except Exception as e:
        print(f"ERROR in get_all_visit_prep_questions: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/submit_intake", methods=["POST"])
def submit_intake():
    """Submit all intake answers at once"""
    try:
        print("DEBUG: submit_intake called")
        data = request.get_json(force=True) or {}
        patient_id = str(data.get("patient_id", "")).upper()
        answers = data.get("answers", {})
        
        print(f"DEBUG: patient_id = {patient_id}")
        print(f"DEBUG: answers = {answers}")
        
        if not patient_id:
            return jsonify({"error": "Missing patient_id"}), 400
        
        patient = get_patient_data(patient_id)
        if not patient:
            return jsonify({"error": "Invalid patient_id"}), 400
            
        # Get or create chat manager for intake
        manager = CHAT_SESSIONS.get(patient_id)
        print(f"DEBUG: manager exists = {manager is not None}")
        
        if not manager:
            print("DEBUG: Creating new ChatManager for intake submission")
            manager = ChatManager(patient_id, patient)
            CHAT_SESSIONS[patient_id] = manager
        
        print(f"DEBUG: intake_session exists = {manager.intake_session is not None}")
        # Store the answers in the intake session
        if manager.intake_session:
            print(f"DEBUG: Storing {len(answers)} answers")
            for qkey, answer in answers.items():
                manager.intake_session.responses[qkey] = answer
            print(f"DEBUG: Total responses now = {len(manager.intake_session.responses)}")
        else:
            print("ERROR: Intake session not found after manager creation")
            return jsonify({"error": "Intake session not found"}), 400
        
        print("DEBUG: Submit intake successful")
        return jsonify({"success": True})
    except Exception as e:
        print(f"ERROR in submit_intake: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/submit_visit_prep", methods=["POST"])
def submit_visit_prep():
    """Submit all visit prep answers at once"""
    try:
        print("DEBUG: submit_visit_prep called")
        data = request.get_json(force=True) or {}
        patient_id = str(data.get("patient_id", "")).upper()
        answers = data.get("answers", {})
        
        print(f"DEBUG: patient_id = {patient_id}")
        print(f"DEBUG: answers = {answers}")
        
        if not patient_id:
            return jsonify({"error": "Missing patient_id"}), 400
        
        patient = get_patient_data(patient_id)
        if not patient:
            return jsonify({"error": "Invalid patient_id"}), 400
            
        # Get or create chat manager for visit prep
        manager = CHAT_SESSIONS.get(patient_id)
        print(f"DEBUG: manager exists = {manager is not None}")
        
        if not manager:
            print("DEBUG: Creating new ChatManager for visit prep submission")
            manager = ChatManager(patient_id, patient, start_at_visit_prep=True)
            CHAT_SESSIONS[patient_id] = manager
        
        # Initialize visit_session if it doesn't exist
        if not manager.visit_session:
            print("DEBUG: Initializing visit_session for submission")
            from visit_prep import VisitSession
            intake_answers = getattr(manager.intake_session, 'responses', {}) if manager.intake_session else {}
            manager.visit_session = VisitSession(patient_id, patient, intake_answers)
        
        print(f"DEBUG: visit_session exists = {manager.visit_session is not None}")
        # Store the answers in the visit session
        print(f"DEBUG: Storing {len(answers)} answers")
        for qkey, answer in answers.items():
            manager.visit_session.responses[qkey] = answer
        print(f"DEBUG: Total responses now = {len(manager.visit_session.responses)}")
        
        print("DEBUG: Submit visit prep successful")
        return jsonify({"success": True})
    except Exception as e:
        print(f"ERROR in submit_visit_prep: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/generate_final_summary", methods=["POST"])
def generate_final_summary():
    try:
        data = request.get_json(force=True) or {}
        patient_id = str(data.get("patient_id", "")).upper()
        
        if not patient_id:
            return jsonify({"error": "Missing patient_id"}), 400
            
        # Get the chat manager for this patient
        manager = CHAT_SESSIONS.get(patient_id)
        if not manager:
            # Try to recreate the manager from stored data
            patient = get_patient_data(patient_id)
            if not patient:
                return jsonify({"error": "Patient not found"}), 400
            manager = ChatManager(patient_id, patient, start_at_visit_prep=True)
            CHAT_SESSIONS[patient_id] = manager
            
        # Generate the final summary
        _, summary = manager.get_next_question()
        
        return jsonify({
            "summary": summary or "No summary available."
        })
    except Exception as e:
        print(f"ERROR in generate_final_summary: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route("/save_comment", methods=["POST"])
def save_comment():
    try:
        data = request.get_json(force=True) or {}
        patient_id = str(data.get("patient_id", "")).upper()
        comment = data.get("comment")
        
        if not patient_id or not comment:
            return jsonify({"error": "Missing patient_id or comment"}), 400
            
        # Get the chat manager for this patient
        manager = CHAT_SESSIONS.get(patient_id)
        if not manager:
            return jsonify({"error": "Session not found"}), 400
            
        # Save the comment to the manager
        manager.save_comment(comment)
        
        return jsonify({"success": True})
    except Exception as e:
        print(f"ERROR in save_comment: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🏥 HEALTHCOACH SERVER STARTING")
    print("="*70)
    print(f"• Environment: {Config.ENV}")
    print(f"• Debug Mode: {Config.DEBUG}")
    print(f"• Port: {Config.PORT}")
    print(f"• Hugging Face Model: {Config.HF_MODEL}")
    print(f"• HF Token: {'Set' if Config.HF_TOKEN else 'Not set (may be required for gated models)'}")
    
    # Model will load on first use (lazy loading)
    print("\n📦 Model Configuration:")
    print("  ✓ LLM will load automatically on first question generation")
    print("  ⏱️  First question may take 2-5 minutes (model downloads on first use)")
    print("  💾 Model will be cached for subsequent questions")
    
    # Show available patients
    print(f"\n👥 Available Patients: {len(PATIENT_DATA)}")
    for pid, name in get_patient_names().items():
        print(f"  • {pid}: {name}")
    
    # Start server
    print("\n" + "="*70)
    print(f"🚀 SERVER READY - Open http://localhost:{Config.PORT} in your browser")
    print("="*70)
    print("\n💡 Watch this terminal for debug output when you use the app\n")
    app.run(host="0.0.0.0", port=Config.PORT, debug=Config.DEBUG)