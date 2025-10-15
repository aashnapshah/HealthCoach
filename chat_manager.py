"""
HealthCoach Chat Flow Manager
Manages the sequential flow of intake verification and visit preparation
"""

from typing import Dict, Optional, Tuple
from enum import Enum
from intake import IntakeSession, get_next_question as get_next_intake_question, generate_verification_summary
from visit_prep import VisitSession, get_next_question as get_next_visit_question, generate_visit_summary

class ChatStage(Enum):
    INTAKE = "intake"              # Medical record verification
    VISIT_PREP = "visit_prep"      # Visit preparation
    COMPLETE = "complete"          # All chats completed

class ChatManager:
    """Manages the complete chat flow for a patient"""
    
    def __init__(self, patient_id: str, patient_data: dict, start_at_visit_prep: bool = False):
        self.patient_id = patient_id
        self.patient_data = patient_data
        self.stage = ChatStage.VISIT_PREP if start_at_visit_prep else ChatStage.INTAKE
        
        # Initialize sessions based on starting point
        if start_at_visit_prep:
            self.intake_session = None
            self.visit_session = VisitSession(patient_id, patient_data)
            self.intake_complete = True
            self.visit_complete = False
        else:
            self.intake_session = IntakeSession(patient_id, patient_data)
            self.visit_session = None
            self.intake_complete = False
            self.visit_complete = False
            
        self.intake_summary = None
        self.visit_summary = None

    def get_next_question(self) -> Tuple[Optional[Dict], Optional[str]]:
        """Get the next question and any transition messages"""
        
        if self.stage == ChatStage.INTAKE:
            # Handle intake verification
            question = get_next_intake_question(self.intake_session)
            
            if question is None or self.intake_session.stage.name.lower() == "summary":
                # Intake complete, generate summary
                self.intake_complete = True
                self.intake_summary = generate_verification_summary(self.intake_session)
                
                # Return completion status and summary
                return {
                    "type": "complete",
                    "question": "Medical record review complete!",
                    "section": "verification",
                    "show_summary": True
                }, self.intake_summary
            
            return question, None
            
        elif self.stage == ChatStage.VISIT_PREP:
            # Handle visit preparation
            if not self.visit_session:
                self.visit_session = VisitSession(self.patient_id, self.patient_data)
            
            question = get_next_visit_question(self.visit_session)
            
            if question is None or self.visit_session.stage.name.lower() == "summary":
                # Visit prep complete
                self.visit_complete = True
                self.visit_summary = generate_visit_summary(self.visit_session)
                self.stage = ChatStage.COMPLETE
                
                # Combine both summaries
                combined_summary = f"""MEDICAL RECORD VERIFICATION
{'-' * 50}
{self.intake_summary}

VISIT PREPARATION
{'-' * 50}
{self.visit_summary}"""
                
                return {
                    "type": "complete",
                    "question": "All done! I've prepared a complete summary for your doctor.",
                    "section": "complete"
                }, combined_summary
            
            return question, None
        
        return None, None

    def submit_answer(self, answer: str, question_key: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Submit an answer and get the next question"""
        
        if self.stage == ChatStage.INTAKE:
            if question_key:
                self.intake_session.responses[question_key] = answer
            return self.get_next_question()
            
        elif self.stage == ChatStage.VISIT_PREP:
            if question_key:
                self.visit_session.responses[question_key] = answer
            return self.get_next_question()
        
        return None, None
