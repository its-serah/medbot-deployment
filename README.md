# MedBot - AI Medical Assistant

Production-ready medical chatbot with Flask backend, modern web interface, and Docker deployment. Supports fine-tuned LLM models with intelligent fallback system.

## Features

- **AI-Powered**: Fine-tuned medical LLM support with LoRA adapters
- **Professional UI**: Modern chat interface with real-time validation
- **Production Ready**: Docker containerization with health monitoring
- **Secure**: Non-root execution, input validation, medical disclaimers
- **Deployment Ready**: Multi-platform support (Render, Railway, Fly.io, Heroku)

## Quick Deploy

### Render.com (Recommended)
1. Fork this repository
2. Go to [render.com](https://render.com) → "New Web Service"
3. Connect your GitHub repository
4. Auto-detects Docker → Deploy
5. Live in 3 minutes with HTTPS

### Local Development
```bash
git clone https://github.com/its-serah/medbot-deployment.git
cd medbot-deployment
pip install -r requirements.txt
python app.py
# Visit http://localhost:8080
```

### Docker
```bash
docker build -t medbot .
docker run -p 8080:8080 medbot
```

## Architecture

- **Backend**: Flask with Gunicorn WSGI server
- **Frontend**: Responsive HTML/CSS/JS with chat interface
- **AI**: Transformer models with PEFT LoRA adapters
- **Deployment**: Docker with health checks and monitoring
- **Security**: Input sanitization, medical compliance, error handling

## Model Integration

Place your fine-tuned model files in `/models/` directory:
```python
model.save_pretrained("./models")
tokenizer.save_pretrained("./models")
```

The application automatically detects and loads custom models. Includes intelligent fallback system for reliability.

## Configuration

**Environment Variables:**
- `PORT`: Server port (default: 8080)
- `FLASK_ENV`: production/development
- `HOST`: Bind address (default: 0.0.0.0)

**Health Check:** `GET /health` - Returns service status and model availability

## Medical Compliance

**Disclaimer:** Educational purposes only. Not intended for medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals.

## Technical Stack

- **Python 3.10+** with Flask framework
- **PyTorch** with Transformers and PEFT
- **Docker** with multi-stage builds
- **Gunicorn** production WSGI server
- **Modern web standards** (HTML5, CSS3, ES6)

## License

MIT License - see LICENSE file for details.
