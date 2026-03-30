import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

# Import the classes
from engine.processor import DocumentProcessor
from engine.vector_db import VectorEngine
from engine.agents import LegalAgents

# 1. Setup
load_dotenv()
app = FastAPI(title="Legal AI Suite API", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    mode: str = "lawyer"
    chat_history: str = ""
    search_web: bool = False

# --- HEALTH CHECK (Render uses this to see if you are alive) ---
@app.get("/")
def health_check():
    return {"status": "Live", "service": "Legal AI Suite"}

@app.post("/upload-docs")
async def upload_documents(files: List[UploadFile] = File(...)):
    try:
        # Initializing inside the function for "Speed Boot"
        doc_processor = DocumentProcessor()
        vector_engine = VectorEngine()
        
        docs = doc_processor.process_to_documents(files)
        if not docs:
            raise HTTPException(status_code=400, detail="No readable text found.")
            
        vector_engine.save_documents(docs)
        return {"status": "success", "count": len(docs)}
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(req: QueryRequest):
    try:
        # Initializing inside the function so Render starts fast
        vector_engine = VectorEngine()
        legal_agents = LegalAgents()
        
        retriever = vector_engine.get_retriever()
        
        if req.search_web:
            answer = legal_agents.search_kanoon_direct(retriever, req.question, req.chat_history)
        elif req.mode.lower() == "judge":
            answer = legal_agents.get_judge_response(retriever, req.question, req.chat_history)
        else:
            answer = legal_agents.get_lawyer_response(retriever, req.question, req.chat_history)
            
        return {"answer": answer, "mode": req.mode}
    except Exception as e:
        print(f"CRITICAL QA ERROR: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/timeline")
async def get_timeline():
    try:
        # Initializing inside the function
        vector_engine = VectorEngine()
        legal_agents = LegalAgents()
        
        retriever = vector_engine.get_retriever()
        timeline = legal_agents.generate_timeline(retriever)
        return {"timeline": timeline}
    except Exception as e:
        print(f"CRITICAL TIMELINE ERROR: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # This part handles the Render Port correctly
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)