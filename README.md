# Dialogue Summarizer API

Production-ready REST API for summarizing customer-agent conversations using FLAN-T5-base.

## 🚀 Live API

**Deployed on Hugging Face Spaces (Completely FREE)**

## Features

✅ Zero-shot summarization (no examples)
✅ Few-shot summarization (with examples)
✅ REST API endpoints
✅ Health check endpoint
✅ Interactive API documentation

## API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model": "google/flan-t5-base (local)",
  "version": "2.0",
  "ready": true
}
```

### Summarize Dialogue
```bash
POST /summarize
```

Request:
```json
{
  "dialogue": "Customer: I want to cancel my subscription. Agent: Can I help you find a solution first?",
  "strategy": "few_shot"
}
```

Response:
```json
{
  "summary": "Customer wants to cancel subscription. Agent offered to help.",
  "strategy": "few_shot",
  "status": 200,
  "model": "google/flan-t5-base",
  "dialogue_length": 17
}
```

## Test the API

### Using curl:
```bash
# Health check
curl https://[your-space-url]/health

# Summarize
curl -X POST "https://[your-space-url]/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "dialogue": "Customer: Help me cancel. Agent: Let me assist you.",
    "strategy": "few_shot"
  }'
```

### Using Python:
```python
import requests

url = "https://[your-space-url]/summarize"
data = {
    "dialogue": "Customer: I want to cancel. Agent: Can I help?",
    "strategy": "few_shot"
}

response = requests.post(url, json=data)
print(response.json())
```

## Model

- **Model:** google/flan-t5-base
- **Size:** ~1GB
- **Task:** Text-to-text generation
- **License:** Apache 2.0

## Deployment

Deployed on [Hugging Face Spaces](https://huggingface.co/spaces)

- Auto-deploys from GitHub
- Completely FREE
- No credit card needed

## Author

Your Name - Dialogue Summarizer Portfolio Project

## License

MIT
