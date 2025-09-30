# 🚀 Deployment Guide

## After Adding Data to `training_data.json`

### 1. **Retrain the Model**
```bash
# Option A: Command line
python3 train_model.py train

# Option B: Direct Python
python3 -c "from chatbot import ChatBot; bot = ChatBot(); print(bot.retrain_model())"
```

### 2. **Test Locally**
```bash
# Start the server
python3 app.py

# Test in browser
open http://localhost:8080
```

### 3. **Deploy to Render**

#### **Step 1: Create `render.yaml`**
```yaml
services:
  - type: web
    name: cosmic-ai-chatbot
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: PORT
        value: 8080
```

#### **Step 2: Deploy**
1. Push to GitHub
2. Connect to Render
3. Deploy automatically

## 📁 Clean Project Structure

```
cosmic-ai/
├── app.py                 # Main Flask app
├── chatbot.py            # AI logic
├── train_model.py        # Training script
├── requirements.txt      # Dependencies
├── data/
│   └── training_data.json # Your training data
├── models/               # Generated models
│   ├── chatbot_model.pkl
│   ├── vectorizer.pkl
│   └── label_encoder.pkl
├── templates/
│   └── chat.html         # Chat interface
└── static/               # Empty (for future assets)
```

## 🔄 Workflow

1. **Add data** → Edit `data/training_data.json`
2. **Train model** → Run `python3 train_model.py train`
3. **Test locally** → Run `python3 app.py`
4. **Deploy** → Push to GitHub → Render auto-deploys

## 📊 Training Data Limits

- **Recommended:** 1,000-2,000 entries
- **Maximum:** 5,000 entries
- **File size:** Keep under 500KB
- **Memory usage:** ~150MB for 2,000 entries

## 🛠️ Backend Training Commands

```bash
# Check status
python3 train_model.py status

# Add single Q&A
python3 train_model.py add "Question" "Answer"

# Interactive mode
python3 train_model.py interactive

# Train from file
python3 train_model.py file data/training_data.json
```

## 🌐 API Endpoints

- `GET /` - Chat interface
- `POST /chat` - Send message
- `POST /add_data` - Add training data
- `POST /retrain` - Retrain model
- `GET /get_training_data` - View data

## ✅ Ready for Production!

Your chatbot is now clean, optimized, and ready for deployment! 🎉
