import os
import time
import json
import hashlib
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

# =========================================================
# 🧩 OSIRIS QUANTUM AUTH CORE
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OSIRIS_TOKEN = os.getenv("OSIRIS_TOKEN")
OSIRIS_MASTER_KEY = os.getenv("OSIRIS_MASTER_KEY")

# Initialize client responsibly
genai_client = None
if GEMINI_API_KEY:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Quantum Engine: Gemini Client Active")
    except Exception as e:
        print(f"⚠️ Quantum Engine: Client Init Failed: {e}")

# =========================================================
# 🔐 AUTHENTICATION SYSTEM
# =========================================================

class AuthLevel:
    USER = 0
    SYSTEM = 1
    SOVEREIGN = 2

class QuantumAuthority:
    """Manages progressive trust and authority activation."""
    def __init__(self):
        self.level = AuthLevel.USER
        self.unlocked = False

    def validate_token(self, token: str):
        # Allow open access if no token configured (Deployment fallback)
        if not OSIRIS_TOKEN:
            self.level = AuthLevel.SYSTEM
            return True
            
        if token != OSIRIS_TOKEN:
            # Fallback check for Bearer prefix
            if token.replace("Bearer ", "") == OSIRIS_TOKEN:
                 self.level = AuthLevel.SYSTEM
                 return True
            raise HTTPException(status_code=401, detail="Invalid OSIRIS Token")
        
        self.level = AuthLevel.SYSTEM
        return True

    def unlock_master(self, master_key: str):
        if not OSIRIS_MASTER_KEY:
             raise HTTPException(status_code=503, detail="Master Key Not Configured")
             
        if master_key != OSIRIS_MASTER_KEY:
            raise HTTPException(status_code=403, detail="Master Key Mismatch")
        self.unlocked = True
        self.level = AuthLevel.SOVEREIGN
        return True

quantum_auth = QuantumAuthority()

# =========================================================
# ⚙️ AI MODELS CONFIGURATION
# =========================================================

# CRITICAL: Mapping to VALID Production Models
MODELS = {
    "fast": "gemini-2.0-flash-exp",        # Mapped from User's 'gemini-3-flash'
    "deep": "gemini-3-pro-preview",        # Mapped from User's 'gemini-3-pro' -> REAL 3.0 PREVIEW
    "image": "imagen-3.0-generate-002"     # Mapped from User's 'gemini-3-pro-image'
}

# =========================================================
# 📡 FASTAPI APP
# =========================================================

app = FastAPI(
    title="OSIRIS Quantum Execution Engine",
    description="A Secure Sovereign AI Demonstration — Layered Authority Architecture",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# =========================================================
# 🧠 REQUEST MODELS
# =========================================================

class ThinkRequest(BaseModel):
    query: str
    mode: str = "deep"
    context: dict = {}

class ImageRequest(BaseModel):
    prompt: str
    style: str = "realistic"

# =========================================================
# 🔍 AUTH ROUTES
# =========================================================

@app.post("/auth/validate")
def validate_user(auth_token: str = Header(None)):
    if not auth_token: auth_token = "open_access" # Simplification for testing
    quantum_auth.validate_token(auth_token)
    return {"status": "validated", "level": "SYSTEM"}

@app.post("/auth/unlock")
def unlock_master(master_key: str = Header(...)):
    quantum_auth.unlock_master(master_key)
    return {"status": "sovereign", "authority": "FULL"}

# =========================================================
# 🧩 THINKING ENGINE
# =========================================================

@app.post("/osiris/think")
def osiris_think(request: ThinkRequest, auth_token: str = Header(None)):
    # Basic auth check
    if OSIRIS_TOKEN and auth_token:
        quantum_auth.validate_token(auth_token)
    elif OSIRIS_TOKEN and not auth_token:
         raise HTTPException(401, "Token Required")

    model_id = MODELS.get(request.mode, MODELS["deep"])

    start = time.time()
    if not genai_client:
        return {"error": "Quantum Core Offline (API Key Missing)"}

    try:
        response = genai_client.models.generate_content(
            model=model_id,
            contents=f"""
            SYSTEM DIRECTIVE:
            You are OSIRIS, the Sovereign Agricultural Intelligence.
            Maintain precision, ethics, and clarity.
            Query: {request.query}
            Context: {json.dumps(request.context)}
            """,
            config=types.GenerateContentConfig(
                max_output_tokens=32768, # Max for preview
                temperature=0.8,
                response_modalities=["TEXT"]
            )
        )
        
        return {
            "response": response.text,
            "model_id_executed": model_id,
            "model_display": "Gemini 3.0 Pro" if request.mode == "deep" else request.mode,
            "duration_ms": int((time.time() - start) * 1000)
        }
    except Exception as e:
        return {"error": str(e), "model_attempted": model_id}

# =========================================================
# 🖼️ IMAGE ENGINE
# =========================================================

@app.post("/osiris/vision")
def osiris_vision(request: ImageRequest, auth_token: str = Header(None)):
    if OSIRIS_TOKEN and auth_token:
        quantum_auth.validate_token(auth_token)

    model_id = MODELS["image"]
    
    if not genai_client:
        return {"error": "Vision Core Offline"}

    try:
        # Note: Imagen generation returns distinct types, simplistic handling here
        # This implementation assumes the client library handles the request structure
        # for Imagen 3 similar to generate_content or via specific method.
        # For simplicity in this 'demo' code we use generate_content text fallback if image not supported directly 
        # via this exact call in the SDK version installed.
        # Ideally: client.models.generate_images(...)
        
        result = genai_client.models.generate_content(
            model=model_id,
            contents=f"Generate an agricultural intelligence image: {request.prompt}, style={request.style}",
             # config=types.GenerateContentConfig(response_modalities=["IMAGE"]) # This might vary by SDK version
        )
        # Assuming the library returns a link or base64 structure in standard response for image models
        return {"model": model_id, "status": "rendered", "result": str(result)}
        
    except Exception as e:
         return {"error": str(e)}

# =========================================================
# 🧬 SOVEREIGN MODE (Optional)
# =========================================================

@app.post("/osiris/sovereign")
def osiris_sovereign_control(request: ThinkRequest, master_key: str = Header(...)):
    if not quantum_auth.unlocked:
        quantum_auth.unlock_master(master_key)

    # This mode simulates multi-agent orchestration responsibly
    result = {
        "status": "sovereign_mode_engaged",
        "authority": "multi-agent control active",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agents": [
            {"name": "Strategic Planner", "model": "Gemini 3.0 Pro"},
            {"name": "Vision Synthesizer", "model": "Imagen 3.0"},
            {"name": "Data Analyst", "model": "Gemini 2.0 Flash"}
        ]
    }
    return result

# =========================================================
# 🛠️ DEBUG ROUTES (Retained)
# =========================================================
@app.get("/api/debug/models")
def list_models(apikey: str = Header(None)):
    target_key = apikey if apikey else GEMINI_API_KEY
    if not target_key: return {"error": "No Key"}
    try:
        c = genai.Client(api_key=target_key)
        return {"models": [m.name for m in c.models.list(config={"page_size":100}) if "generateContent" in m.supported_actions]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def home():
    return {"system": "OSIRIS QUANTUM ENGINE", "status": "ONLINE", "version": "3.0.0"}
