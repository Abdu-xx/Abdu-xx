# AI Chatbot — Full Source

A production-ready AI chatbot using Hugging Face Transformers + FastAPI + Google Cloud Run.

## Project Structure

```
ai-chatbot/
├── backend/
│   ├── main.py           # FastAPI app + HuggingFace model
│   ├── requirements.txt  # Python dependencies
│   └── Dockerfile        # For Cloud Run deployment
└── frontend/
    └── index.html        # Chat UI (deploy to Firebase Hosting)
```

---

## Quick Start (Local)

### 1. Run the backend locally

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start the API server
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

The API will be available at http://localhost:8080

### 2. Open the frontend

Open `frontend/index.html` in your browser directly — no build step needed.

---

## Swap to a Better Model

Edit `backend/main.py` line:
```python
MODEL_NAME = os.getenv("MODEL_NAME", "openai-community/gpt2")
```

Replace with any Hugging Face model:

| Model | Size | Quality | RAM needed |
|---|---|---|---|
| `openai-community/gpt2` | 117M | Demo only | 1 GB |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | Good | 4 GB |
| `mistralai/Mistral-7B-Instruct-v0.2` | 7B | Excellent | 16 GB |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 8B | Excellent | 20 GB |

For large models (7B+), use a GPU Cloud Run instance or Vertex AI.

---

## Deploy to Google Cloud

### Backend → Cloud Run

```bash
cd backend

# 1. Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Build and push container
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ai-chatbot-backend

# 3. Deploy to Cloud Run
gcloud run deploy ai-chatbot-backend \
  --image gcr.io/YOUR_PROJECT_ID/ai-chatbot-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --set-env-vars MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Copy the Service URL from the output (e.g. https://ai-chatbot-xxxx.run.app)
```

### Frontend → Firebase Hosting

```bash
cd frontend

# 1. Install Firebase CLI
npm install -g firebase-tools

# 2. Login and init
firebase login
firebase init hosting   # Choose your project, set public dir to "."

# 3. Update API_BASE in index.html:
#    Replace "https://YOUR-CLOUD-RUN-URL" with your Cloud Run service URL

# 4. Deploy
firebase deploy --only hosting
```

---

## API Reference

### POST /chat
```json
{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_new_tokens": 200,
  "temperature": 0.7
}
```
Response:
```json
{
  "reply": "Hello! How can I help you today?",
  "model": "openai-community/gpt2"
}
```

### GET /health
Returns model status and name.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `openai-community/gpt2` | HuggingFace model ID |
| `MAX_NEW_TOKENS` | `200` | Max tokens to generate |
| `TEMPERATURE` | `0.7` | Creativity (0.1–1.0) |
| `SYSTEM_PROMPT` | (built-in) | AI personality/instructions |

---

## Next Steps

- Add **user authentication** with Firebase Auth
- Add **conversation history** persistence with Firestore
- Add **streaming responses** with FastAPI StreamingResponse
- Use **Vertex AI** instead of local model for managed scaling
- Add **rate limiting** with slowapi