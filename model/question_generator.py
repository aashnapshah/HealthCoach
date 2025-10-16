"""
LLM-based Question Generator
Dynamically generates personalized intake questions based on patient's medical history
"""

import sys
import os
import json
import re
import time

print("⏱️  [QUESTION_GENERATOR] Module loading started...")
_module_start = time.time()

# Add model directory to path and import local config
_model_dir = os.path.dirname(os.path.abspath(__file__))
if _model_dir not in sys.path:
    sys.path.insert(0, _model_dir)

# Import ModelConfig from the same directory
print(f"⏱️  [QUESTION_GENERATOR] Loading ModelConfig...")
_config_start = time.time()
try:
    from config import ModelConfig
except ImportError:
    # Fallback if running from different directory
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_config", os.path.join(_model_dir, "config.py"))
    model_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_config_module)
    ModelConfig = model_config_module.ModelConfig
print(f"⏱️  [QUESTION_GENERATOR] ModelConfig loaded ({time.time() - _config_start:.2f}s)")

# LAZY IMPORTS: Heavy ML libraries will be imported only when needed
_ML_IMPORTS_LOADED = False
def _ensure_ml_imports():
    """Import heavy ML libraries only when actually needed"""
    global _ML_IMPORTS_LOADED
    if not _ML_IMPORTS_LOADED:
        print("⏳ [QUESTION_GENERATOR] Loading ML libraries (torch, transformers, langchain)...")
        _ml_start = time.time()
        global HuggingFacePipeline, ChatPromptTemplate, AutoTokenizer, AutoModelForCausalLM, pipeline, torch
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        from langchain.prompts import ChatPromptTemplate
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch as torch_module
        torch = torch_module
        _ML_IMPORTS_LOADED = True
        print(f"✓ [QUESTION_GENERATOR] ML libraries loaded successfully ({time.time() - _ml_start:.2f}s)")

print(f"✓ [QUESTION_GENERATOR] Module loaded (lightweight, ML libraries will load on first use) ({time.time() - _module_start:.2f}s)")

# Global LLM instance for caching
_LOADED_LLM = None

def response_to_text(response) -> str:
    """Normalize an LLM response to plain text."""
    if hasattr(response, 'content'):
        return response.content
    try:
        return str(response)
    except Exception:
        # Fallback to JSON serialization if str() fails
        try:
            return json.dumps(response, default=str)
        except Exception:
            return ""

def extract_json_array(text: str) -> list:
    if not text:
        return []
    # 1) Try direct list
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            questions = parsed.get('questions') or parsed.get('items') or parsed.get('result')
            if isinstance(questions, list):
                return questions
    except Exception:
        pass

    # 2) Try first [...] slice
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    # 2b) Try to find questions: [...] inside an object string
    m2 = re.search(r'"questions"\s*:\s*(\[.*\])', text, re.DOTALL)
    if m2:
        try:
            parsed = json.loads(m2.group(1))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    # 3) Collect top-level JSON objects and wrap them as a list
    objs, depth, start = [], 0, None
    in_str, esc = False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(text[start:i+1])
                start = None
    if objs:
        try:
            return [json.loads(o) for o in objs]
        except Exception:
            return []
    return []

def normalize_questions(raw_list: list, target_section: str) -> list:
    """Coerce question dicts into a consistent schema expected by the UI."""
    normalized = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        question_text = item.get('question') or item.get('prompt') or item.get('text')
        if not isinstance(question_text, str) or not question_text.strip():
            continue
        qtype = item.get('type') or item.get('question_type') or 'yes_no'
        if qtype == 'yes_no_maybe':
            qtype = 'yes_no'
        section = item.get('section') or target_section
        out = {
            'question': question_text.strip(),
            'type': qtype,
            'section': section
        }
        # Passthrough optional fields if present and well-typed
        if isinstance(item.get('options'), list):
            out['options'] = item['options']
        if isinstance(item.get('scale_labels'), dict):
            out['scale_labels'] = item['scale_labels']
        if isinstance(item.get('allow_multiple'), bool):
            out['allow_multiple'] = item['allow_multiple']
        if isinstance(item.get('subsection'), str):
            out['subsection'] = item['subsection']
        normalized.append(out)
    return normalized

def generate_final_summary_llm(patient_data: dict, verification_responses: str, visit_responses: str, llm=None) -> str:
    """Generate a final clinician-ready summary using EHR + verified answers from Parts 1 and 2."""
    # Load heavy ML imports only when actually needed
    _ensure_ml_imports()
    
    medical_record = patient_data.get("medical_record", "")
    patient_name = patient_data.get("name", "Patient")
    patient_demo = f"Age: {patient_data.get('age','')}, Gender: {patient_data.get('gender','')}"

    # Use the provided LLM or initialize a new one
    local_llm = llm
    if local_llm is None:
        local_llm = initialize_question_llm()

    system_text = """You are a clinical documentation assistant. Write a concise, clinically useful summary for the clinician.

GOAL:
- Synthesize the patient's existing EHR with verified updates from Part 1 and visit-prep answers from Part 2.
- Be precise and actionable. Do NOT invent facts. Prefer bullet points.

STRUCTURE (use these section headings):
1) IDENTIFICATION
2) KEY DIAGNOSES & STATUS
3) MEDICATIONS (changes/concerns)
4) ALLERGIES
5) RECENT OBJECTIVE DATA (key labs/vitals/imaging)
6) TODAY'S CONCERNS & GOALS
7) ACTION ITEMS & FOLLOW-UP

STYLE:
- Clear, brief bullets (1–3 lines per bullet). Avoid filler. Include concrete numbers/dates when present.
"""
    system_text_escaped = system_text.replace("{", "{{").replace("}", "}}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text_escaped),
        ("human", (
            "Patient: {patient_name}\n{patient_demo}\n\n"
            "EHR (source context):\n{medical_record}\n\n"
            "Verified updates from Part 1 (verification responses):\n{verification_responses}\n\n"
            "Visit preparation (Part 2) responses:\n{visit_responses}\n\n"
            "Please produce the final summary now."
        ))
    ])

    response = (prompt | local_llm).invoke({
        "patient_name": patient_name,
        "patient_demo": patient_demo,
        "medical_record": medical_record,
        "verification_responses": verification_responses or "",
        "visit_responses": visit_responses or ""
    })

    return response_to_text(response)

def _get_fallback_verification_questions() -> list:
    """Return generic fallback questions when LLM is not available"""
    return [
        {
            "question": "Is all the information in your medical record up to date?",
            "type": "yes_no",
            "section": "verification",
            "key": "ehr_verify_0",
            "context": "Ensuring your records are current"
        },
        {
            "question": "Are you currently taking all medications listed in your chart?",
            "type": "yes_no",
            "section": "verification",
            "key": "ehr_verify_1",
            "context": "Medication verification for safety"
        },
        {
            "question": "Have any of your allergies changed since your last visit?",
            "type": "yes_no",
            "section": "verification",
            "key": "ehr_verify_2",
            "context": "Allergy information must be current"
        },
        {
            "question": "Are all your current medical conditions listed correctly?",
            "type": "yes_no",
            "section": "verification",
            "key": "ehr_verify_3",
            "context": "Accurate diagnosis information"
        },
        {
            "question": "Have you had any recent tests or procedures not shown in your record?",
            "type": "yes_no",
            "section": "verification",
            "key": "ehr_verify_4",
            "context": "Keeping test results complete"
        }
    ]

def initialize_question_llm():
    """Initialize the LLM for question generation using Hugging Face or RunPod"""
    global _LOADED_LLM
    print("\n=== LLM Initialization ===")
    
    # Return cached instance if available
    if _LOADED_LLM is not None:
        print("✓ Using cached LLM instance")
        return _LOADED_LLM
    
    inference_mode = ModelConfig.INFERENCE_MODE.lower()
    print(f"⚡ Inference Mode: {inference_mode}")
    
    # Option 1: Use RunPod API (no local model loading)
    if inference_mode == 'runpod':
        print("  • Using RunPod API endpoint")
        print(f"  • Endpoint ID: {ModelConfig.RUNPOD_ENDPOINT_ID}")
        
        try:
            # Import the RunPod wrapper
            from runpod_llm import RunPodLLM
            
            _LOADED_LLM = RunPodLLM(
                endpoint_id=ModelConfig.RUNPOD_ENDPOINT_ID,
                api_key=ModelConfig.RUNPOD_API_KEY,
                max_new_tokens=ModelConfig.MAX_NEW_TOKENS,
                temperature=ModelConfig.TEMPERATURE
            )
            print("✓ RunPod LLM initialized successfully")
            return _LOADED_LLM
        except Exception as e:
            print(f"❌ Error initializing RunPod LLM: {e}")
            print("  • Make sure RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID are set")
            import traceback
            traceback.print_exc()
            return None
    
    # Option 2: Load Hugging Face model locally
    else:
        # Load heavy ML imports only when actually needed
        _ensure_ml_imports()
        
        print("  • Loading local Hugging Face model")
        print(f"  • Model: {ModelConfig.HF_MODEL}")
        print(f"  • Device: {ModelConfig.DEVICE}")
        
        try:
            # Try to load from cache first (faster, no network check)
            tokenizer = None
            model = None
            use_local_only = True
            
            try:
                print("  • Checking for cached model...")
                # Load tokenizer from cache
                tokenizer = AutoTokenizer.from_pretrained(
                    ModelConfig.HF_MODEL,
                    cache_dir=ModelConfig.HF_CACHE_DIR,
                    local_files_only=True
                )
                print("  ✓ Found cached tokenizer")
                
                # Load model from cache
                print("  • Loading model from cache...")
                model = AutoModelForCausalLM.from_pretrained(
                    ModelConfig.HF_MODEL,
                    cache_dir=ModelConfig.HF_CACHE_DIR,
                    local_files_only=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map=ModelConfig.DEVICE,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                print("  ✓ Loaded model from cache (no network fetch needed)")
            except Exception as cache_err:
                print(f"  • Model not in cache, downloading from Hugging Face...")
                use_local_only = False
            
            # If not in cache, download from Hugging Face
            if not use_local_only:
                # Load tokenizer
                print("  • Downloading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    ModelConfig.HF_MODEL,
                    token=ModelConfig.HF_TOKEN,
                    cache_dir=ModelConfig.HF_CACHE_DIR
                )
                
                # Load model
                print("  • Downloading model (this may take several minutes)...")
                model = AutoModelForCausalLM.from_pretrained(
                    ModelConfig.HF_MODEL,
                    token=ModelConfig.HF_TOKEN,
                    cache_dir=ModelConfig.HF_CACHE_DIR,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map=ModelConfig.DEVICE,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                print("  ✓ Model downloaded and cached for future use")
            
            # Set pad token if not available
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Verify model was loaded correctly
            print(f"  ✓ Model loaded: {model.config.model_type if hasattr(model, 'config') else 'unknown'}")
            print(f"  ✓ Model name: {ModelConfig.HF_MODEL}")
            print(f"  ✓ Device: {model.device if hasattr(model, 'device') else 'distributed'}")
            print(f"  ✓ Using CUDA: {torch.cuda.is_available()}")
            
            # Create pipeline
            print("  • Creating text generation pipeline...")
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=ModelConfig.MAX_NEW_TOKENS,
                temperature=ModelConfig.TEMPERATURE,
                do_sample=True if ModelConfig.TEMPERATURE > 0 else False,
                top_p=0.95,
                repetition_penalty=1.15,
                return_full_text=False
            )
            
            # Wrap in LangChain
            _LOADED_LLM = HuggingFacePipeline(
                pipeline=pipe,
                model_id=ModelConfig.HF_MODEL
            )
            
            print(f"✓ LLM initialized successfully: {ModelConfig.HF_MODEL}")
            return _LOADED_LLM
        except Exception as e:
            print(f"❌ Error initializing LLM: {e}")
            print("  • Falling back to non-LLM mode")
            import traceback
            traceback.print_exc()
            return None  # Graceful fallback

def generate_ehr_verification_questions(patient_data: dict, llm=None) -> list:
    """Generate specific verification questions from EHR data for Part 1"""
    print("\n=== Generating EHR Verification Questions ===")
    patient_name = patient_data.get("name", "Patient")
    print(f"📋 Processing for patient: {patient_name}")
    
    # Extract medical record
    medical_record = patient_data.get("medical_record", "")
    record_preview = medical_record[:100] + "..." if medical_record else "Empty"
    print(f"📝 Medical record preview: {record_preview}")
    
    # Initialize LLM (LAZY LOADING - only when needed)
    print("\n🤖 LLM Status:")
    if llm is None:
        print("  • No LLM provided, initializing new instance (this may take a moment)...")
        llm = initialize_question_llm()
        if llm is None:
            print("  ⚠️  LLM initialization failed - returning fallback questions")
            return _get_fallback_verification_questions()
    else:
        print("  • Using provided LLM instance")
    
    print("\n📤 Preparing prompt template...")
    system_text = """You are a medical assistant creating PRECISE verification questions strictly grounded in the patient's EHR.

OBJECTIVE:
- Confirm critical facts and close gaps that impact care. Do NOT invent details. Use concrete data (names, doses, dates, values).

MAKE EXACTLY 5 QUESTIONS covering:
1) Key diagnoses (most important/active)
2) Medications (name + dose + schedule)
3) Allergies (substance + reaction)
4) Recent objective data (most recent lab/vital/imaging value + date where present)
5) Recent procedures/ER visits/hospitalizations or care plan elements

QUESTION STRUCTURE (each item):
- question: A direct yes/no confirmation tied to a specific fact (include the fact!)
- type: "yes_no"
- section: "verification"
- context: One short plain-English reason why confirming this matters

OUTPUT FORMAT (JSON array ONLY):
[
  {
    "question": "Your record lists Type 2 Diabetes diagnosed in 2018. Is this still correct?",
    "type": "yes_no",
    "section": "verification",
    "context": "Confirms current conditions for treatment planning"
  },
  {
    "question": "You take Metformin 500 mg twice daily. Are you still taking it as prescribed?",
    "type": "yes_no",
    "section": "verification",
    "context": "Medication accuracy is crucial for safety and effectiveness"
  },
  {
    "question": "Your chart shows a penicillin allergy (rash). Is this still accurate?",
    "type": "yes_no",
    "section": "verification",
    "context": "Allergy verification prevents adverse reactions"
  }
]

RULES:
- EXACTLY 5 questions. Each must reference a concrete fact from the EHR.
- Keep language friendly and concise. Return ONLY valid JSON (no extra text)."""
    system_text_escaped = system_text.replace("{", "{{").replace("}", "}}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text_escaped),
        ("human", "Here is the medical record to analyze:\n\n{medical_record}")
    ])
    
    try:
        print("\n🚀 Invoking LLM chain...")
        print("--------------------------------")
        print("Prompt:")
        print("--------------------------------")
        print(prompt)
        print("--------------------------------")
        print("LLM:")
        print("--------------------------------")
        print(llm)
        print("--------------------------------")
        chain = prompt | llm
        
        # Time the inference
        import time
        start_time = time.time()
        print(f"⏱️  Starting generation at {time.strftime('%H:%M:%S')}")
        
        response = chain.invoke({"medical_record": medical_record})
        
        elapsed_time = time.time() - start_time
        print(f"⏱️  Generation completed in {elapsed_time:.2f} seconds")
        print(response)
        print("✓ LLM response received")
        
        # Normalize response to text
        response_text = response_to_text(response)
        
        print("\n📥 Processing LLM Response:")
        print(f"  • Raw response length: {len(response_text)} characters")
        preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
        print(f"  • Preview: {preview}")
        
        # Extract JSON from response
        print("\n🔍 Extracting JSON...")
        questions = extract_json_array(response_text)
        if questions:
            questions = normalize_questions(questions, target_section="verification")
            print(f"✓ Successfully parsed JSON with {len(questions)} questions")

            # Validate questions structure
            print("\n✨ Validating questions structure:")
            valid_questions = []
            for i, q in enumerate(questions):
                if not isinstance(q, dict):
                    print(f"  ❌ Question {i} is not a dict: {q}")
                    continue
                if 'question' not in q:
                    print(f"  ❌ Question {i} missing 'question' field: {q}")
                    continue
                q['key'] = f"ehr_verify_{i}"
                valid_questions.append(q)
                print(f"  ✓ Question {i}: {q['question'][:50]}...")
            
            print(f"\n🎉 Generated {len(valid_questions)} valid questions")
            return valid_questions
                
        else:
            print("❌ No JSON array found in response")
            print("  • No questions will be shown")
            return []
            
    except Exception as e:
        print("\n❌ Error generating questions:")
        print(f"  • {str(e)}")
        import traceback
        print("\n📋 Traceback:")
        traceback.print_exc()
        print("\n❌ LLM failed completely - no questions will be shown")
        return []


def generate_visit_preparation_questions_with_llm(patient_data: dict, llm=None, intake_answers: dict = None) -> list:
    """Generate visit preparation questions with subsections a, b, c, d"""
    
    medical_record = patient_data.get("medical_record", "")
    patient_name = patient_data.get("name", "Patient")
    
    # Use cached LLM if none provided
    if llm is None:
        llm = initialize_question_llm()
    
    # Prepare recent verification answers context
    answers_formatted = "\n".join([f"- {k}: {v}" for k, v in (intake_answers or {}).items()])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical intake assistant. Generate PRECISE, PATIENT-SPECIFIC questions for TODAY'S VISIT PREPARATION using the EHR and the verified updates from Part 1. Do NOT invent data.

SECTION: VISIT PREPARATION
a) Patient priorities for today's visit
b) Current symptoms and concerns they want to discuss
c) Lifestyle factors that affect their health
d) What makes their symptoms better or worse

QUESTION STRUCTURE (each item):
For EACH question, specify:
- question: The question text (patient-friendly language)
- type: One of [yes_no, scale_1_10, multiple_choice, text, yes_no_maybe]
- section: "visit_preparation"
- subsection: "a", "b", "c", or "d"
- If type is "scale_1_10": provide scale_labels with keys "1" and "10"
- If type is "multiple_choice": provide 4-5 options including "Other"
- If type is "yes_no_maybe": use for uncertain situations
- For multiple_choice: set allow_multiple: true for check-all-that-apply questions

OUTPUT FORMAT (JSON):
[
  {
    "question": "What are your main priorities for today's visit?",
    "type": "multiple_choice",
    "section": "visit_preparation",
    "subsection": "a",
    "allow_multiple": true,
    "options": ["Review my medications", "Discuss new symptoms", "Get test results", "Update my treatment plan", "Other"]
  },
  {
    "question": "What symptoms are you currently experiencing?",
    "type": "multiple_choice",
    "section": "visit_preparation",
    "subsection": "b",
    "allow_multiple": true,
    "options": ["Pain", "Fatigue", "Sleep problems", "Digestive issues", "Mood changes", "Other"]
  },
  {
    "question": "How would you rate your current energy level?",
    "type": "scale_1_10",
    "section": "visit_preparation",
    "subsection": "c",
    "scale_labels": {"1": "Very low energy", "10": "High energy"}
  },
  {
    "question": "What activities or situations make your symptoms worse?",
    "type": "multiple_choice",
    "section": "visit_preparation",
    "subsection": "d",
    "allow_multiple": true,
    "options": ["Stress", "Physical activity", "Certain foods", "Weather", "Sleep disruption", "Other"]
  }
]

IMPORTANT:
- Generate 8-10 questions total across all subsections grounded in EHR + verified updates
- Use multiple_choice with allow_multiple: true for check-all-that-apply questions
- Include "Other" option in multiple choice questions (but NOT "Add more")
- NEVER include "Add more" as an option in any multiple choice question
- Use scale_1_10 for rating questions (pain, energy, mood, etc.) with clear labels
- Use yes_no_maybe only when uncertainty is expected
- Focus on actionable planning for today's visit; personalize wording to conditions/meds/answers
- Return ONLY valid JSON, no other text""".replace("{", "{{").replace("}", "}}")),
        ("human", "Here is the medical record to analyze:\n\n{medical_record}\n\nVerified updates from earlier (Part 1):\n{answers}")
    ])
    
    try:
        # Get LLM response
        print("--------------------------------")
        print("Prompt:")
        print("--------------------------------")
        print(prompt)
        print("--------------------------------")
        print("LLM:")
        print("--------------------------------")
        print(llm)
        print("--------------------------------")
        
        chain = prompt | llm
        
        response = chain.invoke({"medical_record": medical_record, "answers": answers_formatted})
        
        # Normalize and extract JSON from response
        response_text = response_to_text(response)
        questions = extract_json_array(response_text)
        if questions:
            questions = normalize_questions(questions, target_section="visit_preparation")
            
            # Add unique keys to each question
            for i, q in enumerate(questions):
                q['key'] = f"visit_prep_{i}"
            
            return questions
        else:
            print("Could not extract JSON from visit preparation response")
            return []
            
    except Exception as e:
        print(f"Error generating visit preparation questions: {e}")
        return []


def generate_confirmation_questions_with_llm(patient_data: dict, llm=None) -> list:
    """Generate questions to confirm EHR data accuracy"""
    
    medical_record = patient_data.get("medical_record", "")
    patient_name = patient_data.get("name", "Patient")
    
    # Use cached LLM if none provided
    if llm is None:
        llm = initialize_question_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical intake assistant. Generate questions to CONFIRM medication adherence and EHR accuracy.

MEDICATION ADHERENCE:
- For EACH medication in their record, create a SEPARATE question asking about frequency
- Use scale_1_10 type with labels: {"1": "Never take it", "10": "Take as prescribed"}
- Be specific - mention the exact medication name and dosage
- Example: "How often are you taking Metformin 500mg twice daily as prescribed?"

OTHER CONFIRMATIONS:
- Allergies: Ask if they have any NEW allergies (yes_no)
- Conditions: Ask if diagnoses are still active (yes_no)

OUTPUT FORMAT (JSON):
[
  {
    "question": "How often are you taking [MEDICATION NAME DOSAGE] as prescribed?",
    "type": "scale_1_10",
    "scale_labels": {"1": "Never take it", "10": "Take as prescribed"}
  },
  {
    "question": "How often are you taking [MEDICATION 2 NAME DOSAGE] as prescribed?",
    "type": "scale_1_10",
    "scale_labels": {"1": "Never take it", "10": "Take as prescribed"}
  },
  {
    "question": "Do you have any NEW allergies to medications or foods that aren't in your record?",
    "type": "yes_no"
  }
]

IMPORTANT:
- Create ONE question PER medication with scale_1_10
- Include exact medication name and dosage from EHR
- Use the scale labels exactly as shown above
- Return ONLY valid JSON, no other text""".replace("{", "{{").replace("}", "}}")),
        ("human", "Here is the medical record to analyze:\n\n{medical_record}")
    ])
    
    try:
        chain = prompt | llm
        response = chain.invoke({"medical_record": medical_record})
        
        response_text = response_to_text(response)
        questions = extract_json_array(response_text)
        if questions:
            questions = normalize_questions(questions, target_section="verification")
            
            for i, q in enumerate(questions):
                q['key'] = f"confirm_{i}"
            
            return questions
        else:
            return []
            
    except Exception as e:
        print(f"Error generating confirmation questions: {e}")
        return []

def generate_visit_preparation_insights(patient_data: dict, llm=None, intake_answers: dict = None) -> list:
    """Generate visit preparation insights and questions based on patient data"""
    
    medical_record = patient_data.get("medical_record", "")
    patient_name = patient_data.get("name", "Patient")
    
    # Use cached LLM if none provided
    if llm is None:
        llm = initialize_question_llm()
    
    answers_formatted = "\n".join([f"- {k}: {v}" for k, v in (intake_answers or {}).items()])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical AI assistant helping patients prepare for their upcoming visit. Generate actionable insights and preparation questions.

ANALYZE THE PATIENT'S MEDICAL RECORD AND GENERATE:
1. KEY INSIGHTS: Important patterns or concerns to discuss
2. PREPARATION QUESTIONS: Questions to help them organize their thoughts
3. ACTION ITEMS: Specific things to bring up with the doctor

OUTPUT FORMAT (JSON):
[
  {
    "type": "insight",
    "category": "medication|symptoms|lifestyle|tests|prevention",
    "insight": "Brief insight about their health",
    "action_item": "What to discuss with doctor",
    "priority": "high|medium|low"
  },
  {
    "type": "preparation_question",
    "question": "Question to help them prepare for the visit",
    "question_type": "text|multiple_choice|yes_no"
  }
]

FOCUS ON:
- Medication adherence and effectiveness
- Symptom patterns and triggers
- Lifestyle factors affecting health
- Recent test results or trends
- Prevention and screening needs
- Questions they should ask their doctor

Keep insights specific and actionable. Return ONLY valid JSON.""".replace("{", "{{").replace("}", "}}")),
        ("human", "Here is the medical record to analyze:\n\n{medical_record}\n\nVerified updates from earlier (Part 1):\n{answers}")
    ])
    
    try:
        # Get LLM response
        chain = prompt | llm
        response = chain.invoke({"medical_record": medical_record, "answers": answers_formatted})
        
        # Extract JSON from response
        response_text = response_to_text(response)
        insights = extract_json_array(response_text)
        if insights:
            return insights
        else:
            print("Could not extract JSON from visit preparation response")
            return []
            
    except Exception as e:
        print(f"Error generating visit preparation insights: {e}")
        return []

def cleanup_llm():
    """Clean up the LLM instance when no longer needed"""
    global _LOADED_LLM
    if _LOADED_LLM is not None:
        try:
            # Close any open connections
            if hasattr(_LOADED_LLM, '_client'):
                _LOADED_LLM._client.close()
            _LOADED_LLM = None
            print("✓ LLM instance cleaned up")
        except Exception as e:
            print(f"❌ Error cleaning up LLM: {e}")

def create_detailed_symptom_questions(symptom: str) -> list:
    """Create focused detailed symptom questions for a specific symptom"""
    
    return [
        {
            "question": f"Tell me more about your {symptom}",
            "type": "text",
            "placeholder": "Describe when it started, how often it happens, what it feels like, and what makes it better or worse"
        },
        {
            "question": f"On a scale of 1-10, how severe is your {symptom}?",
            "type": "scale_1_10",
            "scale_labels": {"1": "Very mild", "10": "Severe"}
        },
        {
            "question": f"Have you tried any treatments for {symptom}?",
            "type": "text",
            "placeholder": "List any medications, home remedies, or treatments you've tried"
        }
    ]