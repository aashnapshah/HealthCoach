"""
RAG setup for patient data
"""

import sys
import os
import time

print("⏱️  [RAG_SETUP] Module loading started...")
_module_load_start = time.time()

# Add data directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))

from config import ModelConfig
from ehr_data import PATIENT_DATA, get_patient_data, get_all_patient_ids

print(f"⏱️  [RAG_SETUP] Basic imports done ({time.time() - _module_load_start:.2f}s)")

# LAZY IMPORTS: Heavy ML libraries will be imported only when needed
_ML_IMPORTS_LOADED = False
def _ensure_ml_imports():
    """Import heavy ML libraries only when actually needed"""
    global _ML_IMPORTS_LOADED
    if not _ML_IMPORTS_LOADED:
        print("⏳ [RAG_SETUP] Loading ML libraries for RAG (torch, transformers, langchain, faiss)...")
        _start = time.time()
        global HuggingFacePipeline, HuggingFaceEmbeddings, AutoTokenizer, AutoModelForCausalLM, pipeline
        global torch, RecursiveCharacterTextSplitter, FAISS, create_retrieval_chain
        global create_stuff_documents_chain, ChatPromptTemplate
        
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch as torch_module
        torch = torch_module
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.prompts import ChatPromptTemplate
        
        _ML_IMPORTS_LOADED = True
        print(f"✓ [RAG_SETUP] ML libraries loaded successfully ({time.time() - _start:.2f}s)")

print(f"✓ [RAG_SETUP] Module loaded ({time.time() - _module_load_start:.2f}s total)")

# Global storage for patient-specific vector stores and chains
patient_vector_stores = {}
patient_retrievers = {}

def initialize_patient_data():
    """Initialize vector stores for all patients"""
    print("\n🔧 [RAG_SETUP] initialize_patient_data() called")
    _start = time.time()
    
    # Load heavy ML imports only when actually needed
    _ensure_ml_imports()
    
    print("  • Initializing embeddings model...")
    _embed_start = time.time()
    # Initialize embeddings using Hugging Face model
    embed_model = HuggingFaceEmbeddings(
        model_name=ModelConfig.EMBEDDING_MODEL,
        cache_folder=ModelConfig.HF_CACHE_DIR
    )
    print(f"  ✓ Embeddings ready ({time.time() - _embed_start:.2f}s)")
    
    # Text splitter using model config
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ModelConfig.CHUNK_SIZE, 
        chunk_overlap=ModelConfig.CHUNK_OVERLAP
    )
    
    # Create vector store for each patient
    print(f"  • Creating vector stores for {len(PATIENT_DATA)} patients...")
    for patient_id, patient_info in PATIENT_DATA.items():
        _patient_start = time.time()
        medical_record = patient_info["medical_record"]
        
        # Split the medical record into chunks
        chunks = text_splitter.split_text(medical_record)
        
        # Create vector store for this patient
        vector_store = FAISS.from_texts(chunks, embed_model)
        patient_vector_stores[patient_id] = vector_store
        
        # Create retriever for this patient
        patient_retrievers[patient_id] = vector_store.as_retriever(search_kwargs={"k": ModelConfig.RETRIEVAL_K})
        print(f"    ✓ Patient {patient_id} ({time.time() - _patient_start:.2f}s)")
    
    print(f"✅ [RAG_SETUP] Initialized EHR data for {len(PATIENT_DATA)} patients ({time.time() - _start:.2f}s total)")

def setup_rag_chain(patient_id: str):
    """Initialize and return the RAG chain for a specific patient"""
    print(f"\n🔧 [RAG_SETUP] setup_rag_chain() called for patient {patient_id}")
    _start = time.time()
    
    # Load heavy ML imports only when actually needed
    _ensure_ml_imports()
    
    # Validate patient ID
    if patient_id.upper() not in patient_retrievers:
        raise ValueError(f"Patient ID {patient_id} not found. Valid IDs: {', '.join(get_all_patient_ids())}")
    
    # Initialize LLM using Hugging Face
    print(f"  • Loading Hugging Face model: {ModelConfig.HF_MODEL}")
    _model_start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(
        ModelConfig.HF_MODEL,
        token=ModelConfig.HF_TOKEN,
        cache_dir=ModelConfig.HF_CACHE_DIR
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        ModelConfig.HF_MODEL,
        token=ModelConfig.HF_TOKEN,
        cache_dir=ModelConfig.HF_CACHE_DIR,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=ModelConfig.DEVICE,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
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
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
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
    
    print(f"✅ [RAG_SETUP] RAG chain ready ({time.time() - _start:.2f}s total)")
    return chain

def get_answer(chain, question: str) -> str:
    """Get an answer from the RAG chain"""
    print(f"\n🔧 [RAG_SETUP] get_answer() called")
    print(f"  • Question: {question[:100]}...")
    _start = time.time()
    try:
        response = chain.invoke({"input": question})
        print(f"  ✓ Answer generated ({time.time() - _start:.2f}s)")
        return response['answer']
    except Exception as e:
        print(f"❌ [RAG_SETUP] Error getting answer from RAG chain: {str(e)}")
        return "I'm sorry, I'm having trouble accessing the medical records right now. Please try again in a moment."