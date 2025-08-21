# 🚀 MedBot Deployment Guide

Your MedBot is ready to deploy! Here are three easy options:

## Option 1: Render.com (Recommended - Easiest)

### Step 1: Push to GitHub
1. Create a new repository on GitHub
2. Push your code:
```bash
git remote add origin https://github.com/YOUR_USERNAME/medbot-deployment.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Render
1. Go to [render.com](https://render.com) and sign up
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Render will auto-detect Docker
5. Set these environment variables:
   - `PORT`: 8080
   - `FLASK_ENV`: production
   - `HOST`: 0.0.0.0
6. Click "Deploy"

**That's it! Your app will be live at: `https://your-app-name.onrender.com`**

---

## Option 2: Railway.app (Also Easy)

1. Go to [railway.app](https://railway.app)
2. Click "Deploy from GitHub repo"
3. Connect your repository
4. Railway auto-detects Docker and deploys
5. Your app will be live with a provided URL

---

## Option 3: Fly.io (Most Powerful)

### Prerequisites
```bash
# Add flyctl to PATH
export PATH="/home/serah/.fly/bin:$PATH"
echo 'export PATH="/home/serah/.fly/bin:$PATH"' >> ~/.bashrc
```

### Deploy Steps
```bash
cd /home/serah/medbot-deployment

# 1. Authenticate (opens browser)
flyctl auth login

# 2. Launch app (uses fly.toml config)
flyctl launch --no-deploy

# 3. Deploy
flyctl deploy
```

Your app will be live at: `https://medbot-ai.fly.dev`

---

## Quick Local Test

Before deploying, test locally:
```bash
./start-docker.sh
# Visit: http://localhost:8080
```

---

## 📋 What's Included

✅ **Dockerfile** - Production-ready container  
✅ **Health checks** - `/health` endpoint for monitoring  
✅ **Security** - Non-root user, environment variables  
✅ **Performance** - Gunicorn WSGI server  
✅ **Fallback system** - Works without trained model  
✅ **Professional UI** - Modern responsive design  

---

## 🔧 After Deployment

### Add Your Model (Optional)
1. Place model files in `/models/` directory
2. Redeploy to load your fine-tuned model

### Monitor Your App
- Health check: `YOUR_URL/health`
- Logs: Check platform dashboard
- Scale: Adjust resources in platform settings

---

## 💡 Recommendations

- **For beginners**: Use Render.com (free tier available)
- **For production**: Use Fly.io (better performance)
- **For quick testing**: Use Railway.app (fastest setup)

Choose the platform that best fits your needs!
