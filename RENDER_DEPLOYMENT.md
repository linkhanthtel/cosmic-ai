# 🚀 Render Deployment Guide

This guide will help you deploy your Cosmic AI application to Render and ensure it works properly in production.

## Prerequisites

1. A GitHub repository with your code
2. A Render account (free tier works)
3. All your code committed and pushed to GitHub

## Step 1: Update Configuration Files

### 1.1 Port Configuration
The app now automatically uses Render's `PORT` environment variable. The code in `app.py` has been updated to:
- Read the `PORT` environment variable (Render provides this automatically)
- Default to port 8080 if not set (for local development)
- Disable debug mode in production

### 1.2 File Handling
File uploads now use `tempfile` for better compatibility with Render's ephemeral filesystem. Files are automatically cleaned up after processing.

## Step 2: Deploy to Render

### Option A: Using render.yaml (Recommended)

1. **Create/Update `render.yaml`** (already created):
```yaml
services:
  - type: web
    name: cosmic-ai
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: PORT
        value: 8080
```

2. **Connect to Render**:
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml` and configure the service

### Option B: Manual Configuration

1. **Create a new Web Service**:
   - Go to Render Dashboard
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure the service**:
   - **Name**: `cosmic-ai` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Plan**: Free tier is sufficient for testing

3. **Environment Variables** (optional, Render sets PORT automatically):
   - `PORT`: 8080 (Render sets this automatically, but you can specify)
   - `PYTHON_VERSION`: 3.11.0 (optional)

## Step 3: Important Considerations

### 3.1 Model Files
Make sure your trained model files are committed to Git:
- `models/chatbot_model.pkl`
- `models/vectorizer.pkl`
- `models/label_encoder.pkl`
- `data/training_data.json`

**Note**: If models don't exist, the chatbot will train on first run using `data/training_data.json`.

### 3.2 File Storage
- **Uploads**: Files are stored temporarily using `tempfile` and automatically deleted after processing
- **Generated Images**: Images are returned as base64 data, not stored on disk
- **No Persistent Storage**: Render's filesystem is ephemeral, so files are lost on restart

### 3.3 Memory and Resources

**Free Tier Limits**:
- 512 MB RAM
- 0.1 CPU
- 100 GB bandwidth/month

**Recommendations**:
- ✅ **AI Chat**: Works perfectly on free tier
- ✅ **File Converter**: Works on free tier (small to medium files)
- ✅ **Document Summarizer**: Works on free tier
- ⚠️ **Image Generator**: 
  - Placeholder images work fine
  - Stable Diffusion requires significant resources (not recommended on free tier)
  - Consider using an API service for image generation in production

### 3.4 Dependencies

The `requirements.txt` includes:
- Core dependencies (required)
- Optional dependencies for advanced features (commented out)

**For production**, you can:
- Keep Stable Diffusion dependencies commented out (saves build time and memory)
- Only uncomment if you have a paid Render plan with more resources

## Step 4: Verify Deployment

1. **Check Build Logs**:
   - Go to your service in Render Dashboard
   - Check "Logs" tab for any errors during build

2. **Test the Application**:
   - Visit your Render URL (e.g., `https://cosmic-ai.onrender.com`)
   - Test each feature:
     - ✅ Home page loads
     - ✅ AI Chat works
     - ✅ File Converter works
     - ✅ Document Summarizer works
     - ✅ Image Generator works (placeholder mode)

3. **Common Issues**:

   **Issue**: "Module not found"
   - **Solution**: Make sure all dependencies are in `requirements.txt`

   **Issue**: "Port already in use"
   - **Solution**: The code now uses `PORT` env variable automatically

   **Issue**: "Model files not found"
   - **Solution**: Make sure model files are committed to Git, or the app will train on first run

   **Issue**: "File upload fails"
   - **Solution**: Check file size limits (50MB max) and ensure tempfile directory is writable

## Step 5: Post-Deployment

### 5.1 Custom Domain (Optional)
- Go to your service settings
- Add a custom domain if desired

### 5.2 Auto-Deploy
- Render automatically deploys on every push to your main branch
- You can disable this in service settings if needed

### 5.3 Monitoring
- Check "Metrics" tab for resource usage
- Monitor "Logs" for any errors

## Troubleshooting

### Build Fails
1. Check build logs for specific errors
2. Verify all dependencies in `requirements.txt` are valid
3. Ensure Python version is compatible (3.8+)

### App Crashes
1. Check runtime logs
2. Verify all required files are in the repository
3. Check memory usage (free tier has 512MB limit)

### Features Not Working
1. **AI Chat**: Ensure `data/training_data.json` exists
2. **File Converter**: Check file size limits
3. **Document Summarizer**: Verify NLTK data is downloaded (happens automatically)
4. **Image Generator**: Placeholder mode should work; Stable Diffusion requires more resources

## Production Recommendations

1. **Use Environment Variables** for sensitive data (API keys, etc.)
2. **Monitor Resource Usage** and upgrade plan if needed
3. **Set up Logging** for better error tracking
4. **Use External Storage** (S3, etc.) for persistent file storage if needed
5. **Consider Caching** for frequently accessed data

## Support

If you encounter issues:
1. Check Render's [documentation](https://render.com/docs)
2. Review build and runtime logs
3. Test locally first to isolate issues
4. Check Render status page for service issues

---

✅ Your app is now configured for Render deployment!

