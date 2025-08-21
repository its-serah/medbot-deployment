#!/bin/bash
# GitHub Repository Setup Script for MedBot Deployment

echo "🐙 Setting up GitHub repository for MedBot deployment..."

# Check if git is configured
if [ -z "$(git config --global user.name)" ]; then
    echo "⚠️  Git not configured. Please run:"
    echo "  git config --global user.name 'Your Name'"
    echo "  git config --global user.email 'your.email@example.com'"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "📁 Current repository status:"
git status

echo ""
echo "📋 Next steps for deployment:"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: medbot-deployment"
echo "   - Make it public (for free deployment)"
echo "   - Don't initialize with README (we have files already)"
echo ""
echo "2. Push to GitHub:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/medbot-deployment.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Deploy using one of these platforms:"
echo "   🟢 Render.com (Easiest): https://render.com"
echo "   🔵 Railway.app (Fast): https://railway.app"
echo "   🟣 Fly.io (Powerful): Follow DEPLOY.md instructions"
echo ""
echo "📖 See DEPLOY.md for detailed deployment instructions!"

# Show files that will be deployed
echo ""
echo "📦 Files ready for deployment:"
find . -name "*.py" -o -name "*.txt" -o -name "*.yml" -o -name "*.yaml" -o -name "*.toml" -o -name "*.md" -o -name "Dockerfile" -o -name "*.sh" | head -20
