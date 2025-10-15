"""
LLM-based Question Generator
Dynamically generates personalized intake questions based on patient's medical history
"""

from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
import json
import re
from rag_setup import OLLAMA_BASE_URL, OLLAMA_MODEL

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

def initialize_question_llm():
    """Initialize the LLM for question generation"""
    global _LOADED_LLM
    print("\n=== LLM Initialization ===")
    
    # Return cached instance if available
    if _LOADED_LLM is not None:
        print("✓ Using cached LLM instance")
        return _LOADED_LLM
        
    print("⚡ Creating new LLM instance:")
    print(f"  • Base URL: {OLLAMA_BASE_URL}")
    print(f"  • Model: {OLLAMA_MODEL}")
    
    try:
        # Create new instance with timeout and deterministic JSON output
        _LOADED_LLM = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            timeout=30,  # 30 second timeout
            temperature=0,
            format='json'
        )
        print("✓ LLM initialized successfully")
        return _LOADED_LLM
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        print("  • Falling back to non-LLM mode")
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
    
    # Initialize LLM
    print("\n🤖 LLM Status:")
    if llm is None:
        print("  • No LLM provided, initializing new instance")
        llm = initialize_question_llm()
    else:
        print("  • Using provided LLM instance")
    
    print("\n📤 Preparing prompt template...")
    system_text = """You are a medical assistant creating specific verification questions from a patient's medical record.

Generate 4-6 specific verification questions that present actual information from their EHR and ask them to confirm or deny it.

QUESTION STRUCTURE:
For EACH question, specify:
- question: Present specific information and ask for confirmation
- type: "yes_no" 
- section: "verification"
- context: Brief explanation of why this matters

OUTPUT FORMAT (JSON):
[
  {
    "question": "Your medical record shows you have diabetes (Type 2). Is this still correct?",
    "type": "yes_no",
    "section": "verification",
    "context": "This helps us ensure your treatment plan is accurate"
  },
  {
    "question": "You are currently taking Metformin 500mg twice daily for diabetes. Are you still taking this medication?",
    "type": "yes_no", 
    "section": "verification",
    "context": "Medication changes are important for your care"
  },
  {
    "question": "Your record shows an allergy to penicillin. Is this still accurate?",
    "type": "yes_no",
    "section": "verification", 
    "context": "Allergy information is critical for safe treatment"
  }
]

IMPORTANT:
- Generate EXACTLY 3 specific questions based on their actual medical data
- Present real information from their record (conditions, medications, allergies, lab values)
- Ask them to confirm if each specific piece of information is still accurate
- Use simple, patient-friendly language
- Return ONLY valid JSON, no other text"""
    system_text_escaped = system_text.replace("{", "{{").replace("}", "}}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text_escaped),
        ("human", "Here is the medical record to analyze:\n\n{medical_record}")
    ])
    
    try:
        print("\n🚀 Invoking LLM chain...")
        chain = prompt | llm
        response = chain.invoke({"medical_record": medical_record})
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


def generate_visit_preparation_questions_with_llm(patient_data: dict, llm=None) -> list:
    """Generate visit preparation questions with subsections a, b, c, d"""
    
    medical_record = patient_data.get("medical_record", "")
    patient_name = patient_data.get("name", "Patient")
    
    # Use cached LLM if none provided
    if llm is None:
        llm = initialize_question_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical intake assistant. Generate questions for TODAY'S VISIT PREPARATION organized into subsections:

SECTION: VISIT PREPARATION
a) Patient priorities for today's visit
b) Current symptoms and concerns they want to discuss
c) Lifestyle factors that affect their health
d) What makes their symptoms better or worse

QUESTION STRUCTURE:
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
- Generate 6-8 questions total across all subsections
- Use multiple_choice with allow_multiple: true for check-all-that-apply questions
- Include "Other" option in multiple choice questions (but NOT "Add more")
- NEVER include "Add more" as an option in any multiple choice question
- Use scale_1_10 for rating questions (pain, energy, mood, etc.)
- Use yes_no_maybe for uncertain situations
- Focus on visit preparation, NOT medication verification (that's covered in verification)
- Make questions specific to the patient's medical conditions
- Return ONLY valid JSON, no other text""".replace("{", "{{").replace("}", "}}")),
        ("human", "Here is the medical record to analyze:\n\n{medical_record}")
    ])
    
    try:
        # Get LLM response
        chain = prompt | llm
        response = chain.invoke({"medical_record": medical_record})
        
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

def generate_visit_preparation_insights(patient_data: dict, llm=None) -> list:
    """Generate visit preparation insights and questions based on patient data"""
    
    medical_record = patient_data.get("medical_record", "")
    patient_name = patient_data.get("name", "Patient")
    
    # Use cached LLM if none provided
    if llm is None:
        llm = initialize_question_llm()
    
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
        ("human", "Here is the medical record to analyze:\n\n{medical_record}")
    ])
    
    try:
        # Get LLM response
        chain = prompt | llm
        response = chain.invoke({"medical_record": medical_record})
        
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