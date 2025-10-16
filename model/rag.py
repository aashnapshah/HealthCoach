from rag_setup import initialize_patient_data, setup_rag_chain, get_answer
from ehr_data import get_patient_names, get_patient_data

# Initialize patient data
print("Initializing patient EHR data...")
initialize_patient_data()
print("\n" + "="*60)
print("🏥 MEDICAL EHR CHATBOT - CLI Version")
print("="*60)

# Display available patients
print("\nAvailable Patients:")
patient_names = get_patient_names()
for pid, name in patient_names.items():
    patient_info = get_patient_data(pid)
    print(f"  {pid}: {name} ({patient_info['age']}yo, {patient_info['gender']})")

# Select patient
print("\n" + "-"*60)
while True:
    patient_id = input("\nEnter Patient ID (or 'quit' to exit): ").strip().upper()
    
    if patient_id.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        exit()
    
    if patient_id in patient_names:
        patient_info = get_patient_data(patient_id)
        print(f"\n✅ Selected: {patient_info['name']} ({patient_id})")
        print(f"   Age: {patient_info['age']} | Gender: {patient_info['gender']}")
        break
    else:
        print(f"❌ Invalid Patient ID. Please choose from: {', '.join(patient_names.keys())}")

# Initialize RAG chain for selected patient
print(f"\nLoading medical records for {patient_info['name']}...")
chain = setup_rag_chain(patient_id)

# Interactive chatbot loop
print("\n" + "="*60)
print(f"💬 HealthCoach - Chat about {patient_info['name']}'s Health")
print("="*60)
print("\n🌟 I'm here to help you understand medical information in simple terms!")
print("\nYou can ask about:")
print("  - What medications you're taking and why")
print("  - What your lab results mean")
print("  - Your medical conditions explained simply")
print("  - Visit notes and doctor's recommendations")
print("\n💡 Tip: Ask questions naturally, like you're talking to a health educator")
print("\nType 'quit', 'exit', or 'q' to stop")
print("Type 'switch' to change patient\n")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break
    
    if user_input.lower() == 'switch':
        # Restart the script
        print("\n" + "="*60 + "\n")
        import sys
        import os
        os.execv(sys.executable, ['python'] + sys.argv)
    
    if not user_input:
        continue
    
    response = chain.invoke({"input": user_input})
    print(f"\nHealthCoach: {response['answer']}\n")
