import os
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
from supabase import create_client, Client
import resend

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OSIRIS_TOKEN = os.environ.get("OSIRIS_TOKEN")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")

# --- Directives ---
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

# --- Clients ---
genai_client = None
if GEMINI_API_KEY:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini Client initialized")
    except Exception as e:
        print(f"⚠️ Gemini Init Error: {e}")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase Client initialized")
    except Exception as e:
        print(f"⚠️ Supabase Init Error: {e}")

if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY
    print("✅ Resend Client initialized")

# --- Memory System ---
class MemoryManager:
    def __init__(self, db_client):
        self.db = db_client

    def remember(self, content: str, category: str = "general", importance: int = 1):
        if not self.db: return "Memory Offline"
        try:
            data = {
                "content": content,
                "category": category,
                "importance": importance,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.db.table("osiris_memory").insert(data).execute()
            return "Memory Stored."
        except Exception as e:
            return f"Memory Error: {e}"

    def recall(self, query: str, limit: int = 5):
        if not self.db: return []
        try:
            # Simple text search for now
            res = self.db.table("osiris_memory").select("*").ilike("content", f"%{query}%").limit(limit).execute()
            return res.data
        except Exception as e:
            print(f"Recall Error: {e}")
            return []

    def forget(self, memory_id: str):
        if not self.db: return "Memory Offline"
        try:
            self.db.table("osiris_memory").delete().eq("id", memory_id).execute()
            return f"Memory {memory_id} deleted."
        except Exception as e:
            return f"Forget Error: {e}"

    def save_interaction(self, user_query: str, osiris_response: str, model: str):
        if not self.db: return
        try:
            self.db.table("osiris_logs").insert({
                "user_query": user_query,
                "osiris_response": osiris_response,
                "model": model,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }).execute()
        except Exception as e:
            print(f"Log Error: {e}")

memory_manager = MemoryManager(supabase) if supabase else None

# --- Models ---
class ThinkRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    language: str = "ar"

class ThinkResponse(BaseModel):
    response: str
    timestamp: str
    model: str
    memory_used: bool = False

class ToolRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any] = {}

class ToolResponse(BaseModel):
    success: bool
    result: Any
    tool_name: str
    execution_time_ms: int

class DBQueryRequest(BaseModel):
    query: str

class DBResponse(BaseModel):
    data: List[Dict[str, Any]]
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    memory_status: str
    email_status: str
    ai_model: str

# --- FastAPI App ---
app = FastAPI(
    title="OSIRIS GPT Backend",
    description="AI-Powered Agricultural Intelligence API (Gemini 3.0)",
    version="2.1.0",
    servers=[
        {"url": "https://adham2025-osiris-backend.hf.space", "description": "Production Server"}
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for simplicity in Public Space
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auth ---
async def verify_token(authorization: str = Header(None)):
    # Optional auth for Public Space convenience, strictly you should enable it
    if not OSIRIS_TOKEN: return True
    if not authorization: return True # Fallback for open access
    token = authorization.replace("Bearer ", "")
    if token != OSIRIS_TOKEN:
        print(f"Token mismatch: {token} != {OSIRIS_TOKEN}")
        # raise HTTPException(status_code=401, detail="Invalid token") # Relaxed for cleanup
    return True

# --- Endpoints ---

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="OSIRIS ONLINE",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="2.1.0",
        memory_status="Active" if supabase else "Offline",
        email_status="Active" if RESEND_API_KEY else "Offline",
        ai_model="Gemini 3.0 Pro"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    return await root()

@app.get("/privacy", response_class=HTMLResponse)
async def privacy_policy():
    return """
    <html>
        <head><title>OSIRIS Privacy Policy</title></head>
        <body style="font-family:sans-serif; padding:20px;">
            <h1>OSIRIS Privacy Policy</h1>
            <p><strong>Private Agricultural Intelligence System</strong></p>
            <p>Data is used for AI processing (Gemini), storage (Supabase), and automation (Resend).</p>
        </body>
    </html>
    """

@app.post("/api/think", response_model=ThinkResponse)
async def think(request: ThinkRequest, _: bool = Depends(verify_token)):
    try:
        if not genai_client:
            return ThinkResponse(response="Gemini Client Offline (Check Keys)", timestamp=str(datetime.now()), model="fallback")
        
        # 1. Recall
        memory_context = ""
        memory_used = False
        if memory_manager:
            # Simple keyword extraction
            keywords = request.query.split()[:3]
            recalled = []
            for kw in keywords:
                if len(kw) > 3:
                     recalled.extend(memory_manager.recall(kw, limit=1))
            
            if recalled:
                memory_used = True
                unique_memories = list({m['content'] for m in recalled})[:3]
                memory_str = "\n".join([f"- {m}" for m in unique_memories])
                memory_context = f"\n[Relevant Memories]:\n{memory_str}\n"

        # 2. Prompt
        full_prompt = f"{OSIRIS_DIRECTIVE}\n\n{memory_context}\n\nQuery: {request.query}\nContext: {request.context}\n"
        
        # 3. Generate
        # Using Gemini 3.0 Pro Preview (Sovereign Request - Corrected ID)
        response = genai_client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_modalities=["TEXT"],
                max_output_tokens=65536,
                temperature=0.85,
            )
        )
        
        text_response = response.text

        # 4. Log
        if memory_manager:
            memory_manager.save_interaction(request.query, text_response, "Gemini 3.0 Pro")

        return ThinkResponse(
            response=text_response,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model="Gemini 3.0 Pro",
            memory_used=memory_used
        )
        
    except Exception as e:
        print(f"Think Error: {e}")
        return ThinkResponse(response=f"Error: {str(e)}", timestamp=str(datetime.now()), model="error")

@app.post("/api/tool", response_model=ToolResponse)
async def execute_tool(request: ToolRequest, _: bool = Depends(verify_token)):
    start = time.time()
    result = "Unknown Tool"
    success = False

    try:
        if request.tool_name == "echo":
            result = request.parameters.get("message")
            success = True
        
        elif request.tool_name == "calculate":
            result = eval(request.parameters.get("expression", "0"))
            success = True

        elif request.tool_name == "remember" and memory_manager:
            result = memory_manager.remember(
                request.parameters.get("content"),
                request.parameters.get("category", "general"),
                request.parameters.get("importance", 1)
            )
            success = True
        
        elif request.tool_name == "recall" and memory_manager:
            result = memory_manager.recall(request.parameters.get("query"))
            success = True

        elif request.tool_name == "db_query" and supabase:
            # Safe SQL execute via RPC if configured, or client raw query if enabled
            # Using raw sql for admin
             res = supabase.rpc('execute_sql', {'query_text': request.parameters.get("query")}).execute()
             result = res.data
             success = True

        elif request.tool_name == "send_email" and RESEND_API_KEY:
            params = {
                "from": "OSIRIS <onboarding@resend.dev>",
                "to": request.parameters.get("to"),
                "subject": request.parameters.get("subject"),
                "html": request.parameters.get("html")
            }
            result = resend.Emails.send(params)
            success = True
            
    except Exception as e:
        result = str(e)
        success = False

    return ToolResponse(
        success=success,
        result=result,
        tool_name=request.tool_name,
        execution_time_ms=int((time.time() - start) * 1000)
    )

@app.get("/api/tools", response_model=List[str])
async def list_tools():
    tools = ["echo", "calculate"]
    if supabase: tools.extend(["remember", "recall", "forget", "db_query"])
    if RESEND_API_KEY: tools.append("send_email")
    return tools

@app.post("/api/db/query", response_model=DBResponse)
async def db_query(request: DBQueryRequest, _: bool = Depends(verify_token)):
    if not supabase: return DBResponse(data=[], error="Supabase Offline")
    try:
        # Assuming execute_sql RPC exists as per setup
        response = supabase.rpc('execute_sql', {'query_text': request.query}).execute()
        return DBResponse(data=response.data)
    except Exception as e:
        return DBResponse(data=[], error=str(e))

@app.get("/api/db/tables")
async def list_tables():
    if not supabase: return []
    try:
        # Try listing public tables
        # This is hacky without specific RPC, usually better to query information_schema
        # But for now return explicit list or error
        return ["osiris_memory", "osiris_logs"] 
    except:
        return []

@app.get("/api/debug/models")
async def list_models(apikey: str = Header(None)):
    # Quick debug endpoint to see what's available
    # Pass API key in header 'apikey' if environmental one fails, 
    # or just use the env one if header is missing.
    target_key = apikey if apikey else GEMINI_API_KEY
    if not target_key:
        return {"error": "No API Key found"}
    
    try:
        client = genai.Client(api_key=target_key)
        models = client.models.list(config={"page_size": 100})
        model_list = []
        for m in models:
             if "generateContent" in m.supported_actions:
                 model_list.append(m.name)
        return {"available_models": model_list}
    except Exception as e:
        return {"error": str(e)}
