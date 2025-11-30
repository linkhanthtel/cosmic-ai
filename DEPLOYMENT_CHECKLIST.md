# ✅ Render Deployment Checklist

Use this checklist to ensure your app is ready for Render deployment.

## Pre-Deployment

- [ ] **Code is committed to GitHub**
  - All files are committed
  - No sensitive data (API keys, passwords) in code
  - `.gitignore` is properly configured

- [ ] **Required files exist**:
  - [ ] `app.py` (main application)
  - [ ] `requirements.txt` (dependencies)
  - [ ] `render.yaml` (deployment config)
  - [ ] `data/training_data.json` (training data)
  - [ ] `templates/` directory with all HTML files
  - [ ] `models/` directory (optional, will be created if missing)

- [ ] **Model files** (if you have trained models):
  - [ ] `models/chatbot_model.pkl`
  - [ ] `models/vectorizer.pkl`
  - [ ] `models/label_encoder.pkl`
  
  **Note**: If models don't exist, the app will train automatically on first run.

- [ ] **Dependencies checked**:
  - [ ] All required packages in `requirements.txt`
  - [ ] Optional packages (Stable Diffusion) are commented out
  - [ ] No conflicting versions

## Deployment Steps

1. [ ] **Push to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. [ ] **Connect to Render**:
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Blueprint" (if using render.yaml)
   - OR "New +" → "Web Service" (for manual setup)
   - Connect your GitHub repository

3. [ ] **Verify Configuration**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
   - Environment: Python 3
   - Plan: Free tier (or your chosen plan)

4. [ ] **Monitor First Deployment**:
   - Watch build logs for errors
   - Check that all dependencies install correctly
   - Verify the app starts without errors

## Post-Deployment Testing

- [ ] **Home Page**: `/` loads correctly
- [ ] **AI Chat**: `/chat` works, can send messages
- [ ] **File Converter**: `/converter` works, can upload files
- [ ] **Document Summarizer**: `/summarizer` works, can upload documents
- [ ] **Image Generator**: `/image-generator` works, can generate images

## Common Issues & Solutions

### Build Fails
- [ ] Check `requirements.txt` for typos
- [ ] Verify Python version compatibility
- [ ] Check build logs for specific errors

### App Crashes on Start
- [ ] Verify `app.py` uses `PORT` environment variable
- [ ] Check that all required directories exist
- [ ] Review runtime logs for errors

### Features Not Working
- [ ] **AI Chat**: Check `data/training_data.json` exists
- [ ] **File Upload**: Verify tempfile permissions
- [ ] **Models**: Ensure model files are in Git or will be created

### Memory Issues
- [ ] Monitor resource usage in Render dashboard
- [ ] Consider upgrading plan if needed
- [ ] Disable heavy features (Stable Diffusion) on free tier

## Production Optimizations

- [ ] **Environment Variables**: Set any needed env vars in Render dashboard
- [ ] **Custom Domain**: Configure if desired
- [ ] **Auto-Deploy**: Verify auto-deploy is enabled
- [ ] **Monitoring**: Set up logging/alerts if needed

## Quick Test Commands

After deployment, test these endpoints:
```bash
# Home page
curl https://your-app.onrender.com/

# Chat endpoint
curl -X POST https://your-app.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

## Success Criteria

✅ App builds without errors
✅ App starts and responds to requests
✅ All pages load correctly
✅ Core features work (chat, converter, summarizer, image generator)
✅ No memory/resource errors
✅ Logs show no critical errors

---

**Ready to deploy?** Follow the steps above and monitor your first deployment carefully!

