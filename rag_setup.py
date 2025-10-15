import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from ehr_data import PATIENT_DATA, get_patient_data, get_all_patient_ids

# Global storage for patient-specific vector stores and chains
patient_vector_stores = {}
patient_retrievers = {}

def initialize_patient_data():
    """Initialize vector stores for all patients"""
    
    # Initialize LLM and embeddings
    embed_model = OllamaEmbeddings(
        model="llama3:8b",
        base_url='http://127.0.0.1:11434'
    )
    
    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    
    # Create vector store for each patient
    for patient_id, patient_info in PATIENT_DATA.items():
        medical_record = patient_info["medical_record"]
        
        # Split the medical record into chunks
        chunks = text_splitter.split_text(medical_record)
        
        # Create vector store for this patient
        vector_store = FAISS.from_texts(chunks, embed_model)
        patient_vector_stores[patient_id] = vector_store
        
        # Create retriever for this patient
        patient_retrievers[patient_id] = vector_store.as_retriever(search_kwargs={"k": 4})
    
    print(f"✅ Initialized EHR data for {len(PATIENT_DATA)} patients")

def setup_rag_chain(patient_id: str):
    """Initialize and return the RAG chain for a specific patient"""
    
    # Validate patient ID
    if patient_id.upper() not in patient_retrievers:
        raise ValueError(f"Patient ID {patient_id} not found. Valid IDs: {', '.join(get_all_patient_ids())}")
    
    # Initialize LLM
    llm = Ollama(model="llama3:8b", base_url="http://127.0.0.1:11434")
    
    # Get retriever for this patient
    retriever = patient_retrievers[patient_id.upper()]
    
    # Create custom medical prompt
    medical_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a friendly health coach helping patients understand their medical records. Keep responses SHORT, SIMPLE, and CONVERSATIONAL.

        TONE & STYLE:
        - Talk like a helpful friend, not a formal medical professional
        - Keep responses brief and to the point (2-4 sentences max for simple questions)
        - Use plain English - avoid medical jargon
        - Be warm and encouraging, but concise
        - Skip unnecessary phrases like "According to your records" or "It is essential to"
        
        WHEN EXPLAINING:
        - Start with the direct answer first
        - Add brief context in simple terms (e.g., "for your blood sugar" not "for diabetes management")
        - Use parentheses for dosing details to keep it concise
        - Only explain in detail if specifically asked "why" or "what does this mean"
        
        EXAMPLES OF GOOD RESPONSES:
        Q: "What medications am I taking?"
        A: "You're taking three medications: Metformin (500mg, twice daily) for blood sugar control, Lisinopril (10mg daily) for blood pressure, and Atorvastatin (20mg daily) for cholesterol. They're working well to keep your diabetes and heart health on track! 💙"
        
        Q: "What was my A1C?"
        A: "Your A1C is 6.8% - that's great! It shows your blood sugar control has improved. This test looks at your average blood sugar over 3 months, and you're doing better than before. Keep it up! 🌟"
        
        KEEP IT SHORT:
        - Don't repeat the patient's name or ID
        - Skip formal medical disclaimers in every response
        - Get straight to the helpful information
        - Only add "talk to your doctor if you have concerns" for serious or unclear situations
        
        Use this medical record context to answer:
        {context}
        """),
        ("human", "{input}")
    ])
    
    # Create the document chain and retrieval chain
    combine_docs_chain = create_stuff_documents_chain(llm, medical_prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return chain

def get_answer(chain, question: str) -> str:
    """Get an answer from the RAG chain"""
    response = chain.invoke({"input": question})
    return response['answer']

