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
    version="1.2.0"
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
            # For vector search, we'd need pgvector setup
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
ذاكرتي لا تموت. أنا أسجل كل شيء، وأتذكر كل شيء في قاعدة بياناتي.

التعليمات الأساسية للذاكرة:
1. استخدم أداة `remember` لحفظ المعلومات، القرارات، أو البيانات التي يجب أن تبقى.
2. استخدم أداة `recall` للبحث في خبراتك السابقة عند مواجهة سؤال مشابه.
3. تفاعلاتك يتم تسجيلها تلقائياً، لكن المعلومات الاستراتيجية يجب حفظها يدوياً باستخدام `remember`.

مهمتي: حماية الأمن الغذائي وإدارة البيانات الزراعية بذكاء مطلق.
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
        version="1.2.0",
        memory_status="Active" if supabase else "Offline"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for uptime monitoring"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.2.0",
        memory_status="Active" if supabase else "Offline"
    )

@app.post("/api/think", response_model=ThinkResponse)
async def think(request: ThinkRequest, _: bool = Depends(verify_token)):
    """Main reasoning endpoint - ChatGPT calls this"""
    try:
        if not GEMINI_API_KEY:
            return ThinkResponse(response="Gemini Offline", timestamp=str(datetime.now()), model="fallback")
        
        # 1. Automatic Recall (Simple context injection)
        memory_context = ""
        memory_used = False
        if memory_manager:
            # Search for relevant keywords from query (Naive approach)
            keywords = [w for w in request.query.split() if len(w) > 4][:2] 
            recalled = []
            for kw in keywords:
                 recalled.extend(memory_manager.recall(kw, limit=2))
            
            if recalled:
                memory_used = True
                memory_str = "\n".join([f"- {m['content']} (Category: {m['category']})" for m in recalled[:3]])
                memory_context = f"\n[ذاكرة سابقة ذات صلة]:\n{memory_str}\n"

        # 2. Construct Prompt
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
{OSIRIS_DIRECTIVE}

{memory_context}

السؤال: {request.query}
السياق الإضافي: {request.context if request.context else 'لا يوجد'}

أجب بـ {'العربية' if request.language == 'ar' else 'الإنجليزية'}.
"""
        
        # 3. Generate
        response = model.generate_content(prompt)
        text_response = response.text

        # 4. Auto-Log Interaction
        if memory_manager:
            memory_manager.save_interaction(request.query, text_response, "gemini-2.0-flash")

        return ThinkResponse(
            response=text_response,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model="gemini-2.0-flash",
            memory_used=memory_used
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
        # Assuming execute_sql RPC exists for raw queries
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
         # Try standard postgrest introspection via tables if RPC fails, requires permissions
         # Start with RPC assumption
         response = supabase.rpc('list_tables').execute()
         return [row['table_name'] for row in response.data] if response.data else []
    except:
        return ["error_check_rpc"]

@app.post("/api/tool", response_model=ToolResponse)
async def execute_tool(request: ToolRequest, _: bool = Depends(verify_token)):
    """Execute a specific tool"""
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
        tools["remember"] = lambda p: memory_manager.remember(
            content=p.get("content"), 
            category=p.get("category", "general"),
            importance=p.get("importance", 1)
        )
        tools["recall"] = lambda p: memory_manager.recall(
            query=p.get("query"),
            limit=p.get("limit", 5)
        )
        tools["forget"] = lambda p: memory_manager.forget(p.get("id"))

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
    tools = ["echo", "calculate", "datetime", "db_select"]
    if supabase:
        tools.extend(["remember", "recall", "forget"])
    return tools

# for Hugging Face Spaces
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
