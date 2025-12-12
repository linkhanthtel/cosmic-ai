# Cosmic AI 

## Features

- 🤖 **Interactive Chat Interface**: Chat with your trained bot in real-time
- 🎓 **Custom Training**: Add your own question-answer pairs to train the bot
- 📊 **Data Management**: View, add, and delete training data through the web interface
- 🔄 **Model Retraining**: Retrain the model with updated data
- 💾 **Persistent Storage**: Training data and models are saved locally
- 🎨 **Modern UI**: Beautiful, responsive web interface
- 🚀 **Easy Setup**: Simple installation and startup process

## How It Works

The chatbot uses a combination of:
- **TF-IDF Vectorization**: Converts text to numerical features
- **Naive Bayes Classification**: Learns patterns from your training data
- **Cosine Similarity**: Fallback matching for better responses
- **Label Encoding**: Maps answers to numerical labels for training

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

## API Endpoints

- `POST /chat` - Send a message to the chatbot
- `POST /add_data` - Add new training data
- `GET /get_training_data` - Retrieve all training data
- `POST /delete_data` - Delete specific training data
- `POST /retrain` - Retrain the model with current data

## Tips for Better Training

1. **Quality over Quantity**: Focus on high-quality, diverse question-answer pairs
2. **Consistent Formatting**: Use consistent question formats (e.g., always start with "What is...", "How do I...")
3. **Cover Different Topics**: Include various topics and question types
4. **Regular Updates**: Retrain your model after adding new data
5. **Test Regularly**: Chat with your bot to identify gaps in knowledge

## Customization

### Adding More Training Data Programmatically

You can add training data programmatically by modifying the `data/training_data.json` file:

```json
[
  {
    "question": "What is machine learning?",
    "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
    "timestamp": "2024-01-01T12:00:00"
  }
]
```

### Modifying the Model

To use a different machine learning model, modify the `train_model` method in `chatbot.py`:

```python
# Replace MultinomialNB with your preferred model
from sklearn.ensemble import RandomForestClassifier
self.model = RandomForestClassifier(n_estimators=100)
```

## Troubleshooting

### Common Issues

1. **"I haven't been trained yet"**: Add some training data and click "Retrain Model"
2. **Poor responses**: Add more diverse training data and retrain
3. **Import errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
4. **Port already in use**: Change the port in `app.py` (line 95) from 5000 to another port

### Performance Tips

- For better performance with large datasets, consider using more powerful models
- Regular retraining helps maintain accuracy
- Monitor the confidence scores in the response generation

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this chatbot trainer!

---

