"""
HealthCoach Visit Preparation System
Manages the pre-visit workflow to gather current health status and visit goals
"""

import sys
import os

# Add model directory to path (now we're in flask_app/api/, need to go up 2 levels)
current_dir = os.path.dirname(os.path.abspath(__file__))
flask_app_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(flask_app_dir)
sys.path.insert(0, os.path.join(project_root, 'model'))

from typing import Dict, Optional, List
from enum import Enum
from question_generator import (
    initialize_question_llm,
    generate_visit_preparation_questions_with_llm,
    generate_visit_preparation_insights,
    generate_final_summary_llm
)

class QuestionType(Enum):
    YES_NO = "yes_no"
    YES_NO_MAYBE = "yes_no_maybe"
    SCALE_1_10 = "scale_1_10"
    TEXT = "text"
    MULTIPLE_CHOICE = "multiple_choice"

class VisitStage(Enum):
    CURRENT_HEALTH = "current_health"    # Gather current health status
    VISIT_GOALS = "visit_goals"          # Discuss visit priorities
    SUMMARY = "summary"                  # Generate visit summary
    COMPLETE = "complete"                # Session completed

class VisitSession:
    """Manages a patient's visit preparation session"""
    
    def __init__(self, patient_id: str, patient_data: dict, intake_answers: Dict[str, str] = None):
        self.patient_id = patient_id
        self.patient_data = patient_data
        self.stage = VisitStage.CURRENT_HEALTH
        self.responses: Dict[str, str] = {}
        self.intake_answers: Dict[str, str] = intake_answers or {}
        
        # Question management
        self.questions: List[Dict] = []
        self.current_index = 0
        
        # Initialize LLM
        try:
            self.llm = initialize_question_llm()
            self.llm_available = True
        except Exception as e:
            print(f"Warning: LLM initialization failed: {e}")
            self.llm_available = False
            self.llm = None

    def get_visit_questions(self) -> List[Dict]:
        """Get visit preparation questions"""
        if not self.questions:
            try:
                if self.llm_available:
                    questions = generate_visit_preparation_questions_with_llm(self.patient_data, self.llm, self.intake_answers)
                    if questions:
                        self.questions = questions
                        return questions
            except Exception as e:
                print(f"Error generating visit questions: {e}")
            
            # Fallback questions if LLM fails
            self.questions = [
                {
                    "question": "Is there anything specific you'd like to discuss with your doctor?",
                    "type": "text",
                    "context": "This helps us make the most of your visit",
                    "key": "specific_concerns"
                },
                {
                    "question": "What are your main priorities for today's visit?",
                    "type": "multiple_choice",
                    "allow_multiple": True,
                    "options": [
                        "Review my medications",
                        "Discuss new symptoms",
                        "Get test results",
                        "Update my treatment plan",
                        "Discuss lifestyle changes",
                        "Other"
                    ],
                    "key": "visit_priorities"
                },
                {
                    "question": "How would you rate your overall health since your last visit?",
                    "type": "scale_1_10",
                    "scale_labels": {"1": "Much worse", "10": "Much better"},
                    "key": "health_rating"
                },
                
            ]
        return self.questions

def get_next_question(session: VisitSession) -> Optional[Dict]:
    """Get the next question in the visit preparation flow"""
    
    if session.stage == VisitStage.CURRENT_HEALTH:
        questions = session.get_visit_questions()
        
        if session.current_index < len(questions):
            question = questions[session.current_index]
            session.current_index += 1
            return {
                **question,
                "section": "visit_prep"
            }
        else:
            # Move to summary
            session.stage = VisitStage.SUMMARY
            return {
                "type": "info",
                "question": "Thank you! I'll prepare a summary for your doctor...",
                "key": "generating_summary"
            }
    
    return None

def generate_visit_summary(session: VisitSession) -> str:
    """Generate a summary of the visit preparation chat"""
    try:
        if session.llm_available:
            # Generate visit insights
            visit_insights = generate_visit_preparation_insights(session.patient_data, session.llm, session.intake_answers)
            
            # Format responses
            responses = "\n".join([
                f"• {q['question']}\n  → {session.responses.get(q['key'], 'No response')}"
                for q in session.questions
                if q['key'] in session.responses
            ])
            
            # Add insights section
            insights_section = ""
            if visit_insights:
                insights_section = "\n\nKEY INSIGHTS:\n" + "\n".join([
                    f"• {insight['insight']}\n  → Action: {insight['action_item']}"
                    for insight in visit_insights
                    if insight.get('type') == 'insight'
                ])
            
            # Also prepare verification responses from Part 1 if available
            verification_responses = "\n".join([
                f"• {k}: {v}" for k, v in (session.intake_answers or {}).items()
            ])

            # Use LLM to synthesize a final, clinician-ready summary
            final_summary = generate_final_summary_llm(
                session.patient_data,
                verification_responses,
                responses,
                session.llm
            )

            return final_summary

    except Exception as e:
        print(f"Error generating summary: {e}")
        
        # Fallback to basic summary
        all_responses = "\n".join([
            f"Q: {key}\nA: {value}"
            for key, value in session.responses.items()
        ])
        
        return f"""VISIT PREPARATION SUMMARY

Patient: {session.patient_data['name']} (ID: {session.patient_id})
Age: {session.patient_data['age']} | Gender: {session.patient_data['gender']}

RESPONSES:
{all_responses}

Generated by HealthCoach"""
