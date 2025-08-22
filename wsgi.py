#!/usr/bin/env python3
"""
WSGI Entry Point for MedBot Application
======================================

This file provides the WSGI application entry point for production deployment.
"""

import logging
from app import create_app

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create the application instance
application = create_app()

if __name__ == "__main__":
    application.run()
