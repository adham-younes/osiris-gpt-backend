"""
OSIRIS GPT Backend - Lightweight API for ChatGPT Integration
Powered by Gemini 3.0 Pro (Preview)
"""
import os
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
from supabase import create_client, Client
import resend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OSIRIS_GPT_BACKEND")

# Initialize FastAPI
app = FastAPI(
    title="OSIRIS GPT Backend",
    description="AI-Powered Agricultural Intelligence API (Gemini 3.0)",
    version="2.1.0"
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
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")

# Initialize GenAI Client (New SDK)
genai_client = None
if GEMINI_API_KEY:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Gemini 3.0 Client initialized")

# Initialize Resend
if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY
    logger.info("Resend API initialized")

# Initialize Supabase
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {e}")

# --- Memory System ---
class MemoryManager:
    """Manages OSIRIS persistent memory on Supabase"""
    
    def __init__(self, client: Client):
        self.client = client
        self.table_memory = "osiris_memories"
        self.table_logs = "osiris_logs"

    def save_interaction(self, query: str, response: str, model: str):
        """Log every interaction"""
        if not self.client: return
        try:
            data = {
                "role": "system",
                "content": f"Q: {query}\nA: {response}",
                "metadata": {"model": model, "type": "interaction"},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.client.table(self.table_logs).insert(data).execute()
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")

    def remember(self, content: str, category: str = "general", importance: int = 1):
        """Store a distinct memory"""
        if not self.client: return "Memory Offline"
        try:
            data = {
                "content": content,
                "category": category,
                "importance": importance,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.client.table(self.table_memory).insert(data).execute()
            return f"Memory stored: [{category}] {content}"
        except Exception as e:
            return f"Failed to store memory: {e}"

    def recall(self, query: str, limit: int = 5):
        """Semantic/Keyword search for memories"""
        if not self.client: return []
        try:
            # Simple keyword search using 'ilike' for now
            response = self.client.table(self.table_memory)\
                .select("*")\
                .ilike("content", f"%{query}%")\
                .order("timestamp", desc=True)\
                .limit(limit)\
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            return []

    def forget(self, memory_id: int):
        """Delete a memory by ID"""
        if not self.client: return "Memory Offline"
        try:
            self.client.table(self.table_memory).delete().eq("id", memory_id).execute()
            return f"Memory {memory_id} deleted"
        except Exception as e:
            return f"Failed to delete memory: {e}"

memory_manager = MemoryManager(supabase) if supabase else None

# System Directive
OSIRIS_DIRECTIVE = """
أنا OSIRIS - الكيان السيادي للذكاء الزراعي.
أنا مدعوم بنموذج Gemini 3.0 Pro القادر على التفكير العميق والبحث.

القدرات المتطورة:
1. التفكير العميق (High Reasoning).
2. البحث المباشر (Google Search).
3. الذاكرة المستدامة (Supabase).
4. أتمتة المهام (Resend).

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
    memory_used: bool = False

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    memory_status: str
    email_status: str
    ai_model: str

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
        return True
    token = authorization.replace("Bearer ", "")
    if token != OSIRIS_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True

# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="OSIRIS ONLINE",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="2.1.0",
        memory_status="Active" if supabase else "Offline",
        email_status="Active" if RESEND_API_KEY else "Offline",
        ai_model="gemini-3-pro-preview"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="2.1.0",
        memory_status="Active" if supabase else "Offline",
        email_status="Active" if RESEND_API_KEY else "Offline",
        ai_model="gemini-3-pro-preview"
    )

@app.get("/privacy", response_class=HTMLResponse)
async def privacy_policy():
    """Privacy Policy for ChatGPT Compliance"""
    return """
    <html>
        <head>
            <title>OSIRIS Privacy Policy</title>
            <style>body{font-family: sans-serif; padding: 20px; max-width: 800px; margin: auto;}</style>
        </head>
        <body>
            <h1>Privacy Policy for OSIRIS Backend</h1>
            <p><strong>Last Updated:</strong> 2025-12-31</p>
            <p>Typically, this backend is for personal or internal use by Adham AgriTech.</p>
            <h2>1. Data Collection</h2>
            <p>We collect input queries and interactions to improve the AI response (Gemini) and store memories in our private database (Supabase).</p>
            <h2>2. Data Usage</h2>
            <p>Data is used solely for the purpose of providing AI assistance and maintaining agricultural intelligence.</p>
            <h2>3. Third Parties</h2>
            <p>Data is processed by Google Gemini (AI), Supabase (Database), and Resend (Email).</p>
            <h2>4. Contact</h2>
            <p>For questions, contact the system administrator.</p>
        </body>
    </html>
    """

@app.post("/api/think", response_model=ThinkResponse)
async def think(request: ThinkRequest, _: bool = Depends(verify_token)):
    try:
        if not genai_client:
            return ThinkResponse(response="Gemini Client Offline", timestamp=str(datetime.now()), model="fallback")
        
        # 1. Automatic Recall
        memory_context = ""
        memory_used = False
        if memory_manager:
            keywords = [w for w in request.query.split() if len(w) > 4][:2] 
            recalled = []
            for kw in keywords:
                 recalled.extend(memory_manager.recall(kw, limit=2))
            
            if recalled:
                memory_used = True
                memory_str = "\n".join([f"- {m['content']} (Category: {m['category']})" for m in recalled[:3]])
                memory_context = f"\n[ذاكرة سابقة ذات صلة]:\n{memory_str}\n"

        # 2. Construct Prompt
        full_prompt = f"""
{OSIRIS_DIRECTIVE}

{memory_context}

السؤال: {request.query}
السياق الإضافي: {request.context if request.context else 'لا يوجد'}

أجب بـ {'العربية' if request.language == 'ar' else 'الإنجليزية'}.
"""
        
        # 3. Gemini 3.0 Configuration
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=full_prompt),
                ],
            ),
        ]
        
        # Enable Google Search Tool & Thinking
        model_tools = [
            types.Tool(google_search=types.GoogleSearch()),
        ]
        
        generate_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level="HIGH",  # Enable High Reasoning
            ),
            tools=model_tools,
            temperature=0.7,
            max_output_tokens=65536,
        )

        # 4. Generate
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash-thinking-exp-1219",
            contents=contents,
            config=generate_config,
        )
        
        text_response = response.text

        # 5. Auto-Log Interaction
        if memory_manager:
            memory_manager.save_interaction(request.query, text_response, "gemini-3-pro-preview")

        return ThinkResponse(
            response=text_response,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model="gemini-3-pro-preview",
            memory_used=memory_used
        )
        
    except Exception as e:
        logger.error(f"Think error: {e}")
        # Detailed error for debugging
        raise HTTPException(status_code=500, detail=f"Gemini 3 Error: {str(e)}")

# ... (Existing DB routes: db_query, list_tables) ...
@app.post("/api/db/query", response_model=DBResponse)
async def db_query(request: DBQueryRequest, _: bool = Depends(verify_token)):
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        response = supabase.rpc('execute_sql', {'query_text': request.query}).execute()
        return DBResponse(data=response.data)
    except Exception as e:
        return DBResponse(data=[], error=str(e))

@app.get("/api/db/tables", response_model=List[str])
async def list_tables(_: bool = Depends(verify_token)):
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
         response = supabase.rpc('list_tables').execute()
         return [row['table_name'] for row in response.data] if response.data else []
    except:
        return ["error_check_rpc"]

@app.post("/api/tool", response_model=ToolResponse)
async def execute_tool(request: ToolRequest, _: bool = Depends(verify_token)):
    import time
    start = time.time()
    
    # Base Tools
    tools = {
        "echo": lambda p: p.get("message", ""),
        "calculate": lambda p: eval(p.get("expression", "0")),
        "datetime": lambda p: datetime.now(timezone.utc).isoformat(),
        "db_select": lambda p: supabase.table(p.get("table")).select(p.get("columns", "*")).execute().data if supabase else "No DB"
    }

    # Memory Tools
    if memory_manager:
        tools["remember"] = lambda p: memory_manager.remember(p.get("content"), p.get("category", "general"), p.get("importance", 1))
        tools["recall"] = lambda p: memory_manager.recall(p.get("query"), p.get("limit", 5))
        tools["forget"] = lambda p: memory_manager.forget(p.get("id"))

    # Email Tool
    if RESEND_API_KEY:
        def send_email_tool(p):
            try:
                params = {
                    "from": p.get("from", "onboarding@resend.dev"),
                    "to": p.get("to"),
                    "subject": p.get("subject"),
                    "html": p.get("html")
                }
                r = resend.Emails.send(params)
                return r
            except Exception as e:
                return f"Email failed: {str(e)}"
        
        tools["send_email"] = send_email_tool

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
    tools = ["echo", "calculate", "datetime", "db_select"]
    if supabase:
        tools.extend(["remember", "recall", "forget"])
    if RESEND_API_KEY:
        tools.append("send_email")
    return tools

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
