"""
MedBot Flask Application
========================

A modern medical Q&A chatbot using fine-tuned LLM model.
This implementation provides a clean web interface for medical questions
and uses your actual trained model for generating responses.

Features:
- Modern responsive UI
- Real model integration (with fallback)
- Input validation and error handling
- Docker ready
- Production deployment ready
"""

import os
import logging
from flask import Flask, render_template, request, jsonify
from model import MedBotModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Application factory pattern for better testing and deployment."""
    app = Flask(__name__)
    
    # Initialize the model
    try:
        model = MedBotModel()
        logger.info("MedBot model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        model = None
    
    @app.route('/')
    def index():
        """Render the main page."""
        return render_template('index.html')
    
    @app.route('/ask', methods=['POST'])
    def ask():
        """Handle medical questions via AJAX."""
        try:
            data = request.get_json()
            question = data.get('question', '').strip()
            
            if not question:
                return jsonify({
                    'error': 'Please enter a medical question.'
                }), 400
            
            if len(question) > 1000:
                return jsonify({
                    'error': 'Question is too long. Please keep it under 1000 characters.'
                }), 400
            
            # Get response from model
            if model:
                response = model.generate_response(question)
            else:
                response = "I apologize, but the medical model is currently unavailable. Please try again later or consult a healthcare professional."
            
            return jsonify({
                'response': response,
                'question': question
            })
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return jsonify({
                'error': 'An error occurred while processing your request. Please try again.'
            }), 500
    
    @app.route('/health')
    def health():
        """Health check endpoint for deployment monitoring."""
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None
        })
    
    @app.errorhandler(404)
    def not_found(error):
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return render_template('500.html'), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    
    # Get port from environment variable (for deployment platforms)
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting MedBot on {host}:{port}")
    logger.info("MedBot deployment ready")
    app.run(host=host, port=port, debug=False)
