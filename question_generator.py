"""
LLM-based Question Generator
Dynamically generates personalized intake questions based on patient's medical history
"""

from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
import json
import re

def initialize_question_llm():
    """Initialize the LLM for question generation"""
    return Ollama(model="llama3:8b", base_url="http://127.0.0.1:11434")

def generate_ehr_verification_questions(patient_data: dict, llm) -> list:
    """Generate specific verification questions from EHR data for Part 1"""
    
    medical_record = patient_data.get("medical_record", "")
    patient_name = patient_data.get("name", "Patient")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical assistant creating specific verification questions from a patient's medical record.

Generate 4-6 specific verification questions that present actual information from their EHR and ask them to confirm or deny it.

QUESTION STRUCTURE:
For EACH question, specify:
- question: Present specific information and ask for confirmation
- type: "yes_no" 
- section: "part1"
- context: Brief explanation of why this matters

OUTPUT FORMAT (JSON):
[
  {{
    "question": "Your medical record shows you have diabetes (Type 2). Is this still correct?",
    "type": "yes_no",
    "section": "part1",
    "context": "This helps us ensure your treatment plan is accurate"
  }},
  {{
    "question": "You are currently taking Metformin 500mg twice daily for diabetes. Are you still taking this medication?",
    "type": "yes_no", 
    "section": "part1",
    "context": "Medication changes are important for your care"
  }},
  {{
    "question": "Your record shows an allergy to penicillin. Is this still accurate?",
    "type": "yes_no",
    "section": "part1", 
    "context": "Allergy information is critical for safe treatment"
  }}
]

IMPORTANT:
- Generate EXACTLY 3 specific questions based on their actual medical data
- Present real information from their record (conditions, medications, allergies, lab values)
- Ask them to confirm if each specific piece of information is still accurate
- Use simple, patient-friendly language
- Return ONLY valid JSON, no other text"""),
        ("human", """Patient: {patient_name}
Medical Record: {medical_record}

Generate specific EHR verification questions. Return ONLY the JSON array.""")
    ])
    
    try:
        print("DEBUG: Invoking LLM chain...")
        chain = prompt | llm
        response = chain.invoke({"patient_name": patient_name, "medical_record": medical_record})
        
        # Convert response to string if it's not already
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        print(f"DEBUG: Response text: {response_text}")
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            questions_json = json_match.group()

            
            try:
                questions = json.loads(questions_json)

                
                # Validate questions structure
                for i, q in enumerate(questions):
                    if not isinstance(q, dict):
                        print(f"ERROR: Question {i} is not a dict: {q}")
                        continue
                    if 'question' not in q:
                        print(f"ERROR: Question {i} missing 'question' field: {q}")
                        continue
                    q['key'] = f"ehr_verify_{i}"
                
                return questions
                
            except json.JSONDecodeError as json_err:
                print("ERROR: LLM failed to generate valid JSON - no questions will be shown")
                return []
        else:
            print("ERROR: LLM failed to generate valid JSON - no questions will be shown")
            return []
            
    except Exception as e:
        print(f"ERROR: Exception in generate_ehr_verification_questions: {e}")
        import traceback
        traceback.print_exc()
        print("ERROR: LLM failed completely - no questions will be shown")
        return []


def generate_visit_preparation_questions_with_llm(patient_data: dict, llm) -> list:
    """Generate visit preparation questions with subsections a, b, c, d"""
    
    medical_record = patient_data.get("medical_record", "")
    patient_name = patient_data.get("name", "Patient")
    
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
  {{
    "question": "What are your main priorities for today's visit?",
    "type": "multiple_choice",
    "section": "visit_preparation",
    "subsection": "a",
    "allow_multiple": true,
    "options": ["Review my medications", "Discuss new symptoms", "Get test results", "Update my treatment plan", "Other"]
  }},
  {{
    "question": "What symptoms are you currently experiencing?",
    "type": "multiple_choice",
    "section": "visit_preparation",
    "subsection": "b",
    "allow_multiple": true,
    "options": ["Pain", "Fatigue", "Sleep problems", "Digestive issues", "Mood changes", "Other"]
  }},
  {{
    "question": "How would you rate your current energy level?",
    "type": "scale_1_10",
    "section": "visit_preparation",
    "subsection": "c",
    "scale_labels": {{"1": "Very low energy", "10": "High energy"}}
  }},
  {{
    "question": "What activities or situations make your symptoms worse?",
    "type": "multiple_choice",
    "section": "visit_preparation",
    "subsection": "d",
    "allow_multiple": true,
    "options": ["Stress", "Physical activity", "Certain foods", "Weather", "Sleep disruption", "Other"]
  }}
]

IMPORTANT:
- Generate 6-8 questions total across all subsections
- Use multiple_choice with allow_multiple: true for check-all-that-apply questions
- Include "Other" option in multiple choice questions (but NOT "Add more")
- NEVER include "Add more" as an option in any multiple choice question
- Use scale_1_10 for rating questions (pain, energy, mood, etc.)
- Use yes_no_maybe for uncertain situations
- Focus on visit preparation, NOT medication verification (that's covered in Part 1)
- Make questions specific to the patient's medical conditions
- Return ONLY valid JSON, no other text"""),
        ("human", """Patient Medical Record:
{medical_record}

Generate visit preparation questions organized into subsections a, b, c, d. Return ONLY the JSON array.""")
    ])
    
    try:
        # Get LLM response
        chain = prompt | llm
        response = chain.invoke({"medical_record": medical_record, "patient_name": patient_name})
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            questions_json = json_match.group()
            questions = json.loads(questions_json)
            
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


def generate_confirmation_questions_with_llm(patient_data: dict, llm) -> list:
    """Generate questions to confirm EHR data accuracy"""
    
    medical_record = patient_data.get("medical_record", "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical intake assistant. Generate questions to CONFIRM medication adherence and EHR accuracy.

MEDICATION ADHERENCE:
- For EACH medication in their record, create a SEPARATE question asking about frequency
- Use scale_1_10 type with labels: {{"1": "Never take it", "10": "Take as prescribed"}}
- Be specific - mention the exact medication name and dosage
- Example: "How often are you taking Metformin 500mg twice daily as prescribed?"

OTHER CONFIRMATIONS:
- Allergies: Ask if they have any NEW allergies (yes_no)
- Conditions: Ask if diagnoses are still active (yes_no)

OUTPUT FORMAT (JSON):
[
  {{
    "question": "How often are you taking [MEDICATION NAME DOSAGE] as prescribed?",
    "type": "scale_1_10",
    "scale_labels": {{"1": "Never take it", "10": "Take as prescribed"}}
  }},
  {{
    "question": "How often are you taking [MEDICATION 2 NAME DOSAGE] as prescribed?",
    "type": "scale_1_10",
    "scale_labels": {{"1": "Never take it", "10": "Take as prescribed"}}
  }},
  {{
    "question": "Do you have any NEW allergies to medications or foods that aren't in your record?",
    "type": "yes_no"
  }}
]

IMPORTANT:
- Create ONE question PER medication with scale_1_10
- Include exact medication name and dosage from EHR
- Use the scale labels exactly as shown above
- Return ONLY valid JSON, no other text"""),
        ("human", """Patient Medical Record:
{medical_record}

Generate medication adherence questions (one per medication) plus allergy confirmation. Return ONLY the JSON array.""")
    ])
    
    try:
        chain = prompt | llm
        response = chain.invoke({"medical_record": medical_record})
        
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            questions_json = json_match.group()
            questions = json.loads(questions_json)
            
            for i, q in enumerate(questions):
                q['key'] = f"confirm_{i}"
            
            return questions
        else:
            return []
            
    except Exception as e:
        print(f"Error generating confirmation questions: {e}")
        return []

def generate_visit_preparation_insights(patient_data: dict, llm) -> list:
    """Generate visit preparation insights and questions based on patient data"""
    
    medical_record = patient_data.get("medical_record", "")
    patient_name = patient_data.get("name", "Patient")
    
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

        Keep insights specific and actionable. Return ONLY valid JSON."""),
        ("human", """Patient: {patient_name}
        Medical Record: {medical_record}
        
        Generate visit preparation insights and questions. Return ONLY the JSON array.""")
    ])
    
    try:
        # Get LLM response
        chain = prompt | llm
        response = chain.invoke({"medical_record": medical_record, "patient_name": patient_name})
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            insights_json = json_match.group()
            insights = json.loads(insights_json)
            return insights
        else:
            print("Could not extract JSON from visit preparation response")
            return []
            
    except Exception as e:
        print(f"Error generating visit preparation insights: {e}")
        return []

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

