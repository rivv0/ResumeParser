# 🚀 ML Resume Matcher - GitHub Codespaces Deployment

## Quick Start in Codespaces

### 1. Create Codespace
- Go to your GitHub repository
- Click **Code** → **Codespaces** → **Create codespace on main**
- Wait for the environment to set up (2-3 minutes)

### 2. Run the Application
```bash
# Option 1: Use the startup script
./start_codespace.sh

# Option 2: Manual start
python app.py
```

### 3. Access Your App
- Codespaces will automatically forward port 5009
- Click the popup notification to open in browser
- Or go to **Ports** tab and click the globe icon next to port 5009

## Features Available in Codespace

✅ **Full ML Resume Matcher functionality**
✅ **200+ skills detection across all industries**
✅ **PDF upload and processing**
✅ **AI-powered job matching**
✅ **3 decimal precision scoring**
✅ **Professional dark theme UI**

## Codespace Benefits

- 🆓 **60 hours free per month**
- 🚀 **Pre-configured environment**
- 🔧 **VS Code in browser**
- 📦 **All dependencies auto-installed**
- 🌐 **Automatic port forwarding**
- 💾 **Persistent storage**

## Usage Tips

1. **First Run**: Model download takes ~30 seconds
2. **File Uploads**: Upload PDFs up to 16MB
3. **Performance**: Codespace has good CPU/memory for ML processing
4. **Sharing**: Make port 5009 public to share with others

## Troubleshooting

### Model Loading Issues
```bash
# Manually pre-load the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Port Not Accessible
- Go to **Ports** tab in VS Code
- Right-click port 5009 → **Port Visibility** → **Public**

### Memory Issues
- Restart the Codespace if needed
- The ML model uses ~500MB RAM

## Development

- Edit `app.py` for functionality changes
- Modify `.devcontainer/devcontainer.json` for environment changes
- Use VS Code extensions for Python development

## Sharing Your App

1. Make port 5009 **Public** in Ports tab
2. Share the generated URL with others
3. URL format: `https://[codespace-name]-5009.app.github.dev`

---

**Your ML Resume Matcher is now running in the cloud for FREE! 🎉**