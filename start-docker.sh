#!/bin/bash
# MedBot Docker Startup Script
# Simple alternative to docker-compose for environments where compose has issues

echo "🚀 Starting MedBot Docker Container..."

# Stop and remove existing container if it exists
docker stop medbot 2>/dev/null || true
docker rm medbot 2>/dev/null || true

# Build the image if it doesn't exist or if --build flag is passed
if [ "$1" = "--build" ] || [ -z "$(docker images -q medbot 2> /dev/null)" ]; then
    echo "🔨 Building MedBot Docker image..."
    docker build -t medbot .
fi

# Run the container
echo "📦 Running MedBot container..."
docker run -d \
    --name medbot \
    --restart unless-stopped \
    -p 8080:8080 \
    -e FLASK_ENV=production \
    -e PORT=8080 \
    -e HOST=0.0.0.0 \
    -v "$(pwd)/models:/app/models:ro" \
    medbot

# Wait a moment for startup
echo "⏳ Waiting for MedBot to start..."
sleep 5

# Check if container is running
if docker ps | grep -q medbot; then
    echo "✅ MedBot is running!"
    echo "🌐 Access the application at: http://localhost:8080"
    echo "🏥 Health check endpoint: http://localhost:8080/health"
    
    # Test health endpoint
    if curl -s http://localhost:8080/health > /dev/null; then
        echo "💚 Health check passed!"
    else
        echo "⚠️  Health check failed - container may still be starting"
    fi
else
    echo "❌ Failed to start MedBot container"
    echo "📋 Container logs:"
    docker logs medbot
    exit 1
fi

echo ""
echo "📊 Container status:"
docker ps | grep medbot

echo ""
echo "🛠️  Management commands:"
echo "  docker logs medbot     - View logs"
echo "  docker stop medbot     - Stop container"  
echo "  docker start medbot    - Start container"
echo "  $0 --build             - Rebuild and restart"
