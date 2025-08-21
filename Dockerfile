# Use Python 3.10 slim base image for efficiency
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Copy requirements and install Python dependencies
COPY --chown=app:app requirements.docker.txt .
# Install PyTorch CPU version first
RUN pip install --user --no-cache-dir torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu
# Install other dependencies
RUN pip install --user --no-cache-dir -r requirements.docker.txt

# Add user's pip bin to PATH
ENV PATH="/home/app/.local/bin:${PATH}"

# Copy application files
COPY --chown=app:app . .

# Create models directory for runtime model loading
RUN mkdir -p models

# Set environment variables for Flask
ENV FLASK_APP=app.py \
    FLASK_ENV=production \
    PORT=8080 \
    HOST=0.0.0.0

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Use gunicorn for production deployment
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "--max-requests", "1000", "--max-requests-jitter", "100", "app:create_app()"]
