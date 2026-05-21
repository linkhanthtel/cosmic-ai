from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from datetime import datetime
from chatbot import ChatBot

app = Flask(__name__)

chatbot = ChatBot()

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('templates', exist_ok=True)


@app.route('/')
def index():
    return redirect(url_for('chat_page'))


@app.route('/chat')
def chat_page():
    return render_template('chat.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

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

        success = chatbot.add_training_data(question, answer)

        if success:
            return jsonify({
                'message': 'Training data added successfully',
                'timestamp': datetime.now().isoformat()
            })
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
        return jsonify({'error': 'Failed to delete training data'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        result = chatbot.retrain_model()

        return jsonify({
            'message': 'Model retrained successfully',
            'trained_samples': result['trained_samples'],
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/conversation/summary', methods=['GET'])
def get_conversation_summary():
    try:
        summary = chatbot.get_conversation_summary()
        return jsonify({
            'summary': summary,
            'topics': chatbot.conversation_topics,
            'message_count': len([entry for entry in chatbot.conversation_history if 'user' in entry])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/conversation/clear', methods=['POST'])
def clear_conversation():
    try:
        message = chatbot.clear_conversation()
        return jsonify({
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/conversation/history', methods=['GET'])
def get_conversation_history():
    try:
        return jsonify({
            'history': chatbot.conversation_history[-10:],
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/personality', methods=['GET'])
def get_personality():
    try:
        personality_info = chatbot.get_personality_info()
        return jsonify(personality_info)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/personality/adjust', methods=['POST'])
def adjust_personality():
    try:
        data = request.get_json()
        trait = data.get('trait')
        value = data.get('value')

        if not trait or value is None:
            return jsonify({'error': 'Trait and value are required'}), 400

        result = chatbot.adjust_personality(trait, value)
        return jsonify({
            'message': result,
            'personality': chatbot.get_personality_info()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_ENV') == 'development' or os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)
