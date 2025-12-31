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
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OSIRIS_GPT_BACKEND")

# Initialize FastAPI
app = FastAPI(
    title="OSIRIS GPT Backend",
    description="AI-Powered Agricultural Intelligence API",
    version="1.1.0"
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
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize Supabase
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {e}")

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

class DBQueryRequest(BaseModel):
    query: str

class DBResponse(BaseModel):
    data: List[Dict[str, Any]]
    count: Optional[int] = None
    error: Optional[str] = None

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
        version="1.1.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for uptime monitoring"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.1.0"
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

@app.post("/api/db/query", response_model=DBResponse)
async def db_query(request: DBQueryRequest, _: bool = Depends(verify_token)):
    """Execute a raw SQL query on Supabase (Use with CAUTION)"""
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    
    try:
        # Assuming an RPC function 'execute_sql' exists or we just rely on client capabilities
        # For security, one should use RPC. We will try a hypothetical 'execute_sql' RPC 
        # or fallback to returning error if not set up, as direct raw SQL client-side is limited by RLS usually.
        # But `supabase-py` client with service role key can theoretically do anything.
        # Here we assume standard key.
        response = supabase.rpc('execute_sql', {'query_text': request.query}).execute()
        return DBResponse(data=response.data)
    except Exception as e:
        return DBResponse(data=[], error=str(e))

@app.get("/api/db/tables", response_model=List[str])
async def list_tables(_: bool = Depends(verify_token)):
    """List all public tables in Supabase"""
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        # Try RPC 'list_tables' first
        try:
             response = supabase.rpc('list_tables').execute()
             return [row['table_name'] for row in response.data] if response.data else []
        except:
             # Fallback: maybe just return a static list or error if RPC not def
             return ["error: rpc_list_tables_missing"]
    except Exception as e:
        return [f"error: {str(e)}"]

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
        "db_select": lambda p: supabase.table(p.get("table")).select(p.get("columns", "*")).execute().data if supabase else "Supabase not connected"
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
    tools = ["echo", "calculate", "datetime"]
    if supabase:
        tools.append("db_select")
    return tools

# for Hugging Face Spaces
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
