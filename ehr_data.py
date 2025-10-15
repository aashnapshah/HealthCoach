# Simulated EHR Data for Multiple Patients

PATIENT_DATA = {
    "P001": {
        "name": "John Smith",
        "age": 45,
        "gender": "Male",
        "dob": "March 15, 1979",
        "medical_record": """
        Patient ID: P001
        Name: John Smith
        Age: 45
        Gender: Male
        
        Medical History:
        - Diagnosed with Type 2 Diabetes in 2018
        - Hypertension managed with medication since 2019
        - No known drug allergies
        
        Current Medications:
        - Metformin 500mg twice daily for diabetes
        - Lisinopril 10mg once daily for blood pressure
        - Atorvastatin 20mg once daily for cholesterol
        
        Recent Vitals (Last Visit - 2 weeks ago):
        - Blood Pressure: 128/82 mmHg
        - Heart Rate: 72 bpm
        - Temperature: 98.6°F
        - Weight: 185 lbs
        - Height: 5'10"
        - BMI: 26.5
        
        Lab Results (1 month ago):
        - HbA1c: 6.8% (improved from 7.2%)
        - Fasting Glucose: 125 mg/dL
        - Total Cholesterol: 180 mg/dL
        - LDL: 95 mg/dL
        - HDL: 55 mg/dL
        - Triglycerides: 150 mg/dL
        
        Recent Visits:
        - 10/01/2025: Annual physical exam - Overall health improving, continue current medications
        - 09/15/2025: Follow-up for diabetes management - A1C trending down
        - 08/20/2025: Blood pressure check - Well controlled
        
        Recommendations:
        - Continue current medication regimen
        - Maintain low-carb diet
        - Exercise 30 minutes daily, 5 days per week
        - Monitor blood glucose daily
        - Next appointment in 3 months
        """
    },
    
    "P002": {
        "name": "Sarah Johnson",
        "age": 32,
        "gender": "Female",
        "dob": "July 22, 1992",
        "medical_record": """
        Patient ID: P002
        Name: Sarah Johnson
        Age: 32
        Gender: Female
        
        Medical History:
        - Asthma diagnosed in childhood
        - Seasonal allergies
        - Previous knee surgery (ACL repair) in 2020
        
        Current Medications:
        - Albuterol inhaler as needed for asthma
        - Fluticasone nasal spray for allergies during spring/fall
        - Daily multivitamin
        
        Recent Vitals (Last Visit - 1 week ago):
        - Blood Pressure: 118/75 mmHg
        - Heart Rate: 68 bpm
        - Temperature: 98.4°F
        - Weight: 140 lbs
        - Height: 5'6"
        - BMI: 22.6
        - Oxygen Saturation: 98%
        
        Lab Results (2 months ago):
        - Complete Blood Count: Normal
        - Thyroid Function: Normal
        - Vitamin D: 28 ng/mL (slightly low)
        - Iron: Normal
        
        Recent Visits:
        - 10/07/2025: Routine checkup - Healthy, recommended Vitamin D supplementation
        - 09/22/2025: Asthma control assessment - Well managed with current inhaler
        - 07/10/2025: Knee follow-up - Full recovery from ACL surgery
        
        Allergies:
        - Pollen (seasonal)
        - Penicillin (mild rash)
        
        Recommendations:
        - Add Vitamin D 2000 IU daily supplement
        - Continue using inhaler as needed
        - Maintain active lifestyle
        - Annual physical next year
        """
    },
    
    "P003": {
        "name": "Robert Chen",
        "age": 58,
        "gender": "Male",
        "dob": "November 8, 1966",
        "medical_record": """
        Patient ID: P003
        Name: Robert Chen
        Age: 58
        Gender: Male
        
        Medical History:
        - Coronary artery disease with stent placement in 2022
        - High cholesterol
        - Former smoker (quit 5 years ago)
        - Family history of heart disease
        
        Current Medications:
        - Aspirin 81mg daily (blood thinner)
        - Clopidogrel 75mg daily (antiplatelet)
        - Rosuvastatin 40mg daily (cholesterol)
        - Metoprolol 50mg twice daily (beta blocker)
        
        Recent Vitals (Last Visit - 3 days ago):
        - Blood Pressure: 132/78 mmHg
        - Heart Rate: 64 bpm
        - Temperature: 98.6°F
        - Weight: 195 lbs
        - Height: 5'11"
        - BMI: 27.2
        
        Lab Results (2 weeks ago):
        - Total Cholesterol: 155 mg/dL
        - LDL: 70 mg/dL (at goal)
        - HDL: 48 mg/dL
        - Triglycerides: 185 mg/dL
        - Troponin: Negative
        - BNP: Normal
        
        Recent Visits:
        - 10/11/2025: Cardiology follow-up - Stent functioning well, no chest pain
        - 09/28/2025: Stress test - Passed with good exercise tolerance
        - 08/15/2025: Echocardiogram - Ejection fraction 55% (normal)
        
        Recent Procedures:
        - Cardiac catheterization with stent (2022)
        
        Recommendations:
        - Continue all cardiac medications
        - Cardiac rehab program - completed successfully
        - Low-sodium, heart-healthy diet
        - Walking 30-45 minutes daily
        - Regular cardiology follow-ups every 6 months
        - Monitor for any chest pain or shortness of breath
        """
    },
    
    "P004": {
        "name": "Maria Rodriguez",
        "age": 28,
        "gender": "Female",
        "dob": "May 3, 1996",
        "medical_record": """
        Patient ID: P004
        Name: Maria Rodriguez
        Age: 28
        Gender: Female
        
        Medical History:
        - Currently pregnant (24 weeks gestation)
        - First pregnancy
        - History of migraines (less frequent during pregnancy)
        - No chronic conditions
        
        Current Medications:
        - Prenatal vitamins daily
        - Folic acid 400mcg daily
        - Iron supplement 27mg daily
        
        Recent Vitals (Last Visit - 1 week ago):
        - Blood Pressure: 115/70 mmHg
        - Heart Rate: 78 bpm
        - Temperature: 98.5°F
        - Weight: 152 lbs (pre-pregnancy: 135 lbs)
        - Height: 5'5"
        - Fundal Height: 24 cm (appropriate for gestational age)
        
        Lab Results (1 month ago):
        - Glucose Tolerance Test: Normal (no gestational diabetes)
        - Hemoglobin: 11.5 g/dL (mild anemia - iron supplementation started)
        - Blood Type: O Positive
        - Group B Strep: Pending
        
        Recent Visits:
        - 10/08/2025: Prenatal visit - Baby's heartbeat strong at 145 bpm
        - 09/24/2025: Anatomy ultrasound - All measurements normal
        - 09/10/2025: Glucose screening - Passed
        
        Ultrasound Findings:
        - Fetal growth: Appropriate for gestational age
        - Amniotic fluid: Normal
        - Placenta: Anterior, normal position
        - Estimated due date: January 25, 2026
        
        Recommendations:
        - Continue prenatal vitamins and iron supplement
        - Increase iron-rich foods for mild anemia
        - Gentle exercise (walking, prenatal yoga)
        - Next visit in 2 weeks
        - Plan for glucose recheck at 28 weeks
        - Childbirth education classes recommended
        """
    }
}

def get_patient_data(patient_id):
    """Retrieve patient data by ID"""
    return PATIENT_DATA.get(patient_id.upper())

def get_all_patient_ids():
    """Get list of all patient IDs"""
    return list(PATIENT_DATA.keys())

def get_patient_names():
    """Get mapping of patient IDs to names"""
    return {pid: data["name"] for pid, data in PATIENT_DATA.items()}

