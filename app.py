from flask import Flask, render_template, request, jsonify
import json
import os
from datetime import datetime
from chatbot import ChatBot

app = Flask(__name__)

# Initialize the chatbot
chatbot = ChatBot()

@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get response from chatbot
        response = chatbot.get_response(user_message)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json()
        training_data = data.get('training_data', [])
        
        if not training_data:
            return jsonify({'error': 'No training data provided'}), 400
        
        # Train the chatbot with new data
        result = chatbot.train_model(training_data)
        
        return jsonify({
            'message': 'Training completed successfully',
            'trained_samples': result['trained_samples'],
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add_data', methods=['POST'])
def add_training_data():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        answer = data.get('answer', '').strip()
        
        if not question or not answer:
            return jsonify({'error': 'Both question and answer are required'}), 400
        
        # Add to training data
        success = chatbot.add_training_data(question, answer)
        
        if success:
            return jsonify({
                'message': 'Training data added successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to add training data'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_training_data', methods=['GET'])
def get_training_data():
    try:
        training_data = chatbot.get_training_data()
        return jsonify({
            'training_data': training_data,
            'count': len(training_data)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_data', methods=['POST'])
def delete_training_data():
    try:
        data = request.get_json()
        index = data.get('index')
        
        if index is None:
            return jsonify({'error': 'Index is required'}), 400
        
        success = chatbot.delete_training_data(index)
        
        if success:
            return jsonify({
                'message': 'Training data deleted successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to delete training data'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        # Retrain the model with all current training data
        result = chatbot.retrain_model()
        
        return jsonify({
            'message': 'Model retrained successfully',
            'trained_samples': result['trained_samples'],
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=8080)
