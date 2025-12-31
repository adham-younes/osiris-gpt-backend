# OSIRIS GPT Backend

AI-Powered Agricultural Intelligence API for ChatGPT Integration.

## Quick Start

### Local Development
```bash
pip install -r requirements.txt
python app.py
```

### Deploy to Render
1. Push to GitHub
2. Connect to Render.com
3. Set environment variables:
   - `GEMINI_API_KEY`
   - `OSIRIS_TOKEN`

### ChatGPT Integration
1. Create Custom GPT
2. Add Action → Import OpenAPI
3. Paste `openapi.yaml` content
4. Update server URL

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/think` | POST | Main AI reasoning |
| `/api/tool` | POST | Execute tool |
| `/api/tools` | GET | List tools |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `OSIRIS_TOKEN` | No | API authentication token |
