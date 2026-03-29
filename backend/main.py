import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict
# Import the classes we just built
from engine.processor import DocumentProcessor
from engine.vector_db import VectorEngine
from engine.agents import LegalAgents

# 1. Setup and Security
load_dotenv()
app = FastAPI(title="Legal AI Suite API", version="1.0")

# Enable CORS so your frontend can talk to your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)




class QueryRequest(BaseModel):
    question: str
    mode: str = "lawyer"
    chat_history: str = ""
    search_web: bool = False # We will send the history as a formatted string
# 2. Initialize Engines
doc_processor = DocumentProcessor()
vector_engine = VectorEngine()
legal_agents = LegalAgents()

@app.post("/upload-docs")
async def upload_documents(files: list[UploadFile] = File(...)):
    try:
        # We need to send the actual file streams
        # and keep the filenames for metadata
        docs = doc_processor.process_to_documents(files)
        
        if not docs:
            raise HTTPException(status_code=400, detail="No readable text found.")
            
        vector_engine.save_documents(docs)
        return {"status": "success", "count": len(docs)}
    except Exception as e:
        # This will print the error in your terminal so you can see it!
        print(f"CRITICAL ERROR: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(req: QueryRequest):
    try:
        # 1. ALWAYS get the PDF vault first!
        retriever = vector_engine.get_retriever()
        
        # PATH A: The Indian Kanoon Web Search
        if req.search_web:
            # We now pass the 'retriever' into the Kanoon function!
            answer = legal_agents.search_kanoon_direct(retriever, req.question, req.chat_history)
            return {"answer": answer, "mode": req.mode}
            
        # PATH B: Default PDF Vault Chat
        if req.mode.lower() == "judge":
            answer = legal_agents.get_judge_response(retriever, req.question, req.chat_history)
        else:
            answer = legal_agents.get_lawyer_response(retriever, req.question, req.chat_history)
            
        return {"answer": answer, "mode": req.mode}
    except Exception as e:
        print(f"CRITICAL QA ERROR: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))
# --- THE NEW TIMELINE ENDPOINT ---
@app.post("/timeline")
async def get_timeline():
    try:
        # Grab the PDF vault
        retriever = vector_engine.get_retriever()
        
        # Call our new Timeline Agent
        timeline = legal_agents.generate_timeline(retriever)
        
        return {"timeline": timeline}
    except Exception as e:
        print(f"CRITICAL TIMELINE ERROR: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)