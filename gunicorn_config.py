# Gunicorn configuration file for MedBot deployment
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 8080)}"
backlog = 2048

# Worker processes
workers = 1
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = 'medbot'

# Server mechanics
preload_app = True
