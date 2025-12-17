# Cosmic AI 

## Features

- 🤖 **Interactive Chat Interface**: Chat with your trained bot in real-time
- 🎓 **Custom Training**: Add your own question-answer pairs to train the bot

## Installation

1. **Clone or download this repository**
   ```bash
   cd cosmic-ai
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8080`

## Usage

### 1. Adding Training Data

1. In the "Training Data Management" section, enter a question and its corresponding answer
2. Click "Add Data" to add it to your training dataset
3. Repeat this process to build up your knowledge base

### 2. Training the Model

1. After adding some training data, click "Retrain Model"
2. The system will train the AI model with your data
3. You'll see a success message when training is complete

### 3. Chatting with Your Bot

1. Go to the chat section on the left
2. Type your message and press Enter or click "Send"
3. Your bot will respond based on the training data you provided

### 4. Managing Training Data

- **View**: All your training data is displayed in the right panel
- **Delete**: Click the "Delete" button next to any training item to remove it
- **Update**: Delete old data and add new data to update your bot's knowledge

## File Structure

```
cosmic-ai/
├── app.py                 # Main Flask application
├── chatbot.py            # Chatbot logic and training
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Web interface
├── data/
│   └── training_data.json # Your training data (auto-created)
├── models/               # Trained models (auto-created)
│   ├── chatbot_model.pkl
│   ├── vectorizer.pkl
│   └── label_encoder.pkl
└── README.md
```


