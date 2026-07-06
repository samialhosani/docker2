import uuid
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import load_config, AppConfig
from providers import LLMProviderFactory
from student_profile import get_student_profile, StudentProfile, EnrolledCourse
from chat_manager import ChatDatabase
from rag_manager import RAGManager
from material_manager import MaterialManager, LectureInfo, MaterialFile
from agent import EducationAgent

system_config: AppConfig = None
llm = None
chat_db: ChatDatabase = None
rag_manager: RAGManager = None
material_manager: MaterialManager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifecycle manager (Replaces @app.on_event)"""
    global system_config, llm, chat_db, rag_manager, material_manager
    print("🚀 Starting up API Server...")
    
    system_config = load_config()
    llm = LLMProviderFactory.create_llm(system_config.active_llm_config)
    chat_db = ChatDatabase(system_config.db_path)
    rag_manager = RAGManager(system_config.vector_db_path)
    material_manager = MaterialManager(system_config)
    
    print("✅ All services initialized successfully!")
    yield
    print("🛑 Shutting down API Server...")

app = FastAPI(
    title="Educational AI Assistant API",
    description="API for the Personalized Teaching Assistant Agent",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    student_id: str
    message: str
    course_id: str = "GENERAL"
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    course_id: str
    reply: str

class IngestRequest(BaseModel):
    course_id: str
    course_year: int
    lecture_number: int


@app.get("/health", tags=["System"])
def health_check():
    """Check if the API is running."""
    return {"status": "ok", "mode": system_config.mode}

@app.get("/student/{student_id}", response_model=StudentProfile, tags=["Student"])
def get_student_endpoint(student_id: str):
    """Retrieve the full academic profile, courses, and deadlines for a student."""
    profile = get_student_profile(student_id, db_path=system_config.db_path)
    if not profile:
        raise HTTPException(status_code=404, detail="Student not found.")
    return profile

@app.get("/student/{student_id}/courses", response_model=List[EnrolledCourse], tags=["Student"])
def get_student_courses(student_id: str):
    """Retrieve only the enrolled courses for a student."""
    profile = get_student_profile(student_id, db_path=system_config.db_path)
    if not profile:
        raise HTTPException(status_code=404, detail="Student not found.")
    return profile.enrolled_courses

@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
def chat_with_agent(req: ChatRequest):
    """
    Send a message to the AI Agent. 
    The agent will dynamically decide to use DB queries or RAG based on the prompt.
    """
    profile = get_student_profile(req.student_id, db_path=system_config.db_path)
    if not profile:
        raise HTTPException(status_code=404, detail="Student not found.")
    
    session_id = req.session_id or str(uuid.uuid4())
    
    agent = EducationAgent(
        llm=llm,
        profile=profile,
        db=chat_db,
        rag_manager=rag_manager,
        session_id=session_id
    )
    
    try:
        reply = agent.send_message(req.message, course_id=req.course_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")
        
    return ChatResponse(
        session_id=session_id,
        course_id=req.course_id,
        reply=reply
    )

@app.get("/chat/history/{session_id}", tags=["Agent"])
def get_chat_history(session_id: str, course_id: str = "GENERAL", limit: int = 20):
    """Retrieve past messages for a specific chat session."""
    history = chat_db.get_history(session_id, course_id, limit=limit)
    return {"session_id": session_id, "course_id": course_id, "history": history}

@app.get("/courses/{course_id}/lectures", response_model=List[LectureInfo], tags=["Materials"])
def get_course_lectures(course_id: str):
    """Fetch all available lectures for a given course."""
    lectures = material_manager.get_course_lectures(course_id)
    if not lectures:
        raise HTTPException(status_code=404, detail="No lectures found for this course.")
    return lectures

@app.post("/materials/ingest", tags=["Materials"])
def ingest_lecture_materials(req: IngestRequest):
    """
    Trigger the RAG ingestion pipeline for a specific lecture.
    """
    normalized_course_id = material_manager.normalize_course_id(req.course_id)
    files = material_manager.get_lecture_materials(
        course_year=req.course_year,
        course_id=normalized_course_id,
        lecture_number=req.lecture_number
    )
    
    if not files:
        raise HTTPException(status_code=404, detail="No material files found in the specified directory.")
        
    try:
        rag_manager.ingest_materials(normalized_course_id, files)
        return {"status": "success", "message": f"Successfully ingested {len(files)} files for {normalized_course_id} Lecture {req.lecture_number}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/materials/upload", tags=["Materials"])
async def upload_lecture_materials(
    course_year: int = Form(...),
    course_id: str = Form(...),
    lecture_number: int = Form(...),
    files: List[UploadFile] = File(...)
):
    """Upload lecture materials and ingest them for RAG retrieval."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for upload.")

    normalized_course_id = material_manager.normalize_course_id(course_id)
    saved_files: List[MaterialFile] = []

    for upload in files:
        try:
            content = await upload.read()
            saved_files.append(
                material_manager.save_lecture_material(
                    course_year=course_year,
                    course_id=normalized_course_id,
                    lecture_number=lecture_number,
                    file_name=upload.filename,
                    file_bytes=content,
                )
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save {upload.filename}: {e}")

    try:
        rag_manager.ingest_materials(normalized_course_id, saved_files)
        return {
            "status": "success",
            "message": f"Uploaded and ingested {len(saved_files)} files for {normalized_course_id} Lecture {lecture_number}.",
            "uploaded_files": [file.file_name for file in saved_files],
            "course_id": normalized_course_id,
            "course_year": course_year,
            "lecture_number": lecture_number,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload ingestion failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
