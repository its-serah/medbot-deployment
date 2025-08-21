# MedBot - AI Medical Assistant

A modern, containerized medical Q&A chatbot built with Flask and fine-tuned LLM models. This implementation provides a clean web interface for medical questions and uses trained models for generating responses.

## Features

- 🤖 **AI-Powered**: Uses fine-tuned LLM models for medical Q&A
- 🎨 **Modern UI**: Responsive, professional interface
- 🐳 **Dockerized**: Easy deployment with Docker containers
- 🚀 **Production Ready**: Optimized for cloud deployment
- 📱 **Mobile Friendly**: Responsive design works on all devices
- ⚡ **Fast & Efficient**: Optimized model loading and inference

## Project Structure

```
medbot-deployment/
├── app.py                 # Flask application
├── model.py              # Model wrapper with fallback support
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── templates/           # HTML templates
│   ├── index.html      # Main chat interface
│   ├── 404.html        # Error pages
│   └── 500.html
├── static/             # Static assets
│   ├── style.css      # Modern CSS styling
│   └── script.js      # JavaScript functionality
└── models/            # Directory for model files
```

## Quick Start

### Local Development

1. **Clone and setup**:
   ```bash
   cd medbot-deployment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add your model** (optional):
   - Place your fine-tuned model files in the `models/` directory
   - The app will use fallback responses if no model is found

3. **Run locally**:
   ```bash
   python app.py
   ```
   Visit http://localhost:8080

### Docker Deployment

1. **Build the container**:
   ```bash
   docker build -t medbot .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8080:8080 medbot
   ```

## Cloud Deployment

### Option 1: Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)
2. Choose "Docker" as the SDK
3. Upload all files to your Space repository
4. Your app will be automatically deployed

### Option 2: Render

1. Connect your GitHub repository to [Render](https://render.com)
2. Create a new Web Service
3. Set the following:
   - **Build Command**: `docker build -t medbot .`
   - **Start Command**: `docker run -p $PORT:8080 medbot`
   - **Port**: 8080

### Option 3: Railway

1. Connect to [Railway](https://railway.app)
2. Deploy from GitHub repository
3. Railway will auto-detect the Dockerfile

### Option 4: Fly.io

1. Install flyctl: https://fly.io/docs/getting-started/installing-flyctl/
2. Login: `flyctl auth login`
3. Deploy: `flyctl deploy`

## Model Integration

### Using Your Fine-Tuned Model

1. **Prepare your model files**:
   ```
   models/
   ├── config.json
   ├── tokenizer.json
   ├── tokenizer_config.json
   ├── adapter_model.bin  # If using LoRA
   └── other model files...
   ```

2. **Update model path**: The app automatically searches common paths, or you can specify a custom path in `model.py`

### Supported Model Types

- GPT-based models (GPT-Neo, GPT-NeoX, etc.)
- Models with LoRA adapters
- 4-bit quantized models
- HuggingFace Transformers compatible models

## Configuration

### Environment Variables

- `PORT`: Server port (default: 8080)
- `HOST`: Server host (default: 0.0.0.0)
- `FLASK_ENV`: Flask environment (production/development)

### Model Configuration

Edit `model.py` to adjust:
- Model loading paths
- Generation parameters (temperature, top_p, etc.)
- Fallback responses
- Maximum token limits

## API Endpoints

- `GET /`: Main chat interface
- `POST /ask`: Submit medical questions (JSON)
- `GET /health`: Health check for monitoring

## Medical Disclaimer

⚠️ **Important**: This application is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns.

## Development

### Adding Features

1. **New endpoints**: Add routes in `app.py`
2. **UI changes**: Modify templates and CSS
3. **Model improvements**: Update `model.py`

### Testing

```bash
# Run basic tests
python -m pytest

# Test with Docker
docker build -t medbot-test . && docker run --rm -p 8080:8080 medbot-test
```

## Production Considerations

- **Security**: Uses non-root user in Docker
- **Performance**: Single worker, adjust based on traffic
- **Monitoring**: Health check endpoint included
- **Scaling**: Consider load balancing for high traffic
- **Model Size**: Large models may need GPU or more memory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is provided as-is for educational purposes. Please ensure compliance with your model's license terms and medical AI regulations in your jurisdiction.

## Support

For issues and questions:
1. Check the error logs in your deployment platform
2. Review the fallback responses in `model.py`
3. Ensure model files are correctly formatted
4. Verify all dependencies are installed

---

**Built with ❤️ for medical AI education**
