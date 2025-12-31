from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS just in case
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "OSIRIS ONLINE", "version": "2.1.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/privacy")
def privacy():
    return "Privacy Policy: Minimal Test"

@app.post("/api/think")
def think():
    return {"response": "OSIRIS Think working!"}

@app.post("/api/db/query")
def db():
    return {"response": "OSIRIS DB working!"}
