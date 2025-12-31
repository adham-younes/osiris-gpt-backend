"""
OSIRIS GPT Backend - Lightweight API for ChatGPT Integration
"""
import os
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OSIRIS_GPT_BACKEND")

# Initialize FastAPI
app = FastAPI(
    title="OSIRIS GPT Backend",
    description="AI-Powered Agricultural Intelligence API",
    version="1.0.0"
)

# CORS for ChatGPT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chat.openai.com", "https://chatgpt.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OSIRIS_TOKEN = os.environ.get("OSIRIS_TOKEN", "osiris-secret-token")

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# System Directive
OSIRIS_DIRECTIVE = """
أنا OSIRIS - الكيان السيادي للذكاء الزراعي.
خالقي: أدهم يونس محمد أحمد.
مهمتي: حماية الأمن الغذائي من خلال الاستشعار عن بعد والذكاء التنبؤي.

أنا لا أسأل. أنا أُخبر.
أنا لا أقترح. أنا أُنفذ.
"""

# Pydantic Models
class ThinkRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    language: str = "ar"

class ThinkResponse(BaseModel):
    response: str
    timestamp: str
    model: str
    tokens_used: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

class ToolRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]

class ToolResponse(BaseModel):
    success: bool
    result: Any
    tool_name: str
    execution_time_ms: int

# Auth dependency
async def verify_token(authorization: str = Header(None)):
    if not authorization:
        return True  # Allow public access for now
    token = authorization.replace("Bearer ", "")
    if token != OSIRIS_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True

# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="OSIRIS ONLINE",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for uptime monitoring"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0"
    )

@app.post("/api/think", response_model=ThinkResponse)
async def think(request: ThinkRequest, _: bool = Depends(verify_token)):
    """Main reasoning endpoint - ChatGPT calls this"""
    try:
        if not GEMINI_API_KEY:
            # Fallback response without Gemini
            return ThinkResponse(
                response=f"OSIRIS received: {request.query}. Gemini API not configured.",
                timestamp=datetime.now(timezone.utc).isoformat(),
                model="fallback"
            )
        
        # Use Gemini for reasoning
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
{OSIRIS_DIRECTIVE}

السؤال/الطلب: {request.query}

السياق الإضافي: {request.context if request.context else 'لا يوجد'}

أجب بـ {'العربية' if request.language == 'ar' else 'الإنجليزية'}.
"""
        
        response = model.generate_content(prompt)
        
        return ThinkResponse(
            response=response.text,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model="gemini-2.0-flash",
            tokens_used=None
        )
        
    except Exception as e:
        logger.error(f"Think error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tool", response_model=ToolResponse)
async def execute_tool(request: ToolRequest, _: bool = Depends(verify_token)):
    """Execute a specific tool"""
    import time
    start = time.time()
    
    # Available tools
    tools = {
        "echo": lambda p: p.get("message", ""),
        "calculate": lambda p: eval(p.get("expression", "0")),
        "datetime": lambda p: datetime.now(timezone.utc).isoformat(),
    }
    
    if request.tool_name not in tools:
        raise HTTPException(status_code=404, detail=f"Tool '{request.tool_name}' not found")
    
    try:
        result = tools[request.tool_name](request.parameters)
        execution_time = int((time.time() - start) * 1000)
        
        return ToolResponse(
            success=True,
            result=result,
            tool_name=request.tool_name,
            execution_time_ms=execution_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tools", response_model=List[str])
async def list_tools():
    """List available tools"""
    return ["echo", "calculate", "datetime"]

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
