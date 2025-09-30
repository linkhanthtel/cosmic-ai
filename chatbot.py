import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime

class ChatBot:
    def __init__(self):
        self.training_data_file = 'data/training_data.json'
        self.model_file = 'models/chatbot_model.pkl'
        self.vectorizer_file = 'models/vectorizer.pkl'
        self.label_encoder_file = 'models/label_encoder.pkl'
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.model = None
        self.training_data = []
        
        # Load existing data and model
        self.load_training_data()
        self.load_model()
    
    def load_training_data(self):
        """Load training data from JSON file"""
        if os.path.exists(self.training_data_file):
            try:
                with open(self.training_data_file, 'r', encoding='utf-8') as f:
                    self.training_data = json.load(f)
            except Exception as e:
                print(f"Error loading training data: {e}")
                self.training_data = []
        else:
            self.training_data = []
    
    def save_training_data(self):
        """Save training data to JSON file"""
        os.makedirs(os.path.dirname(self.training_data_file), exist_ok=True)
        try:
            with open(self.training_data_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving training data: {e}")
    
    def load_model(self):
        """Load trained model if it exists"""
        if os.path.exists(self.model_file) and os.path.exists(self.vectorizer_file) and os.path.exists(self.label_encoder_file):
            try:
                self.model = joblib.load(self.model_file)
                self.vectorizer = joblib.load(self.vectorizer_file)
                self.label_encoder = joblib.load(self.label_encoder_file)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
    
    def save_model(self):
        """Save trained model"""
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        try:
            joblib.dump(self.model, self.model_file)
            joblib.dump(self.vectorizer, self.vectorizer_file)
            joblib.dump(self.label_encoder, self.label_encoder_file)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def add_training_data(self, question, answer, include_timestamp=False):
        """Add new training data"""
        try:
            entry = {
                'question': question,
                'answer': answer
            }
            if include_timestamp:
                entry['timestamp'] = datetime.now().isoformat()
            
            self.training_data.append(entry)
            self.save_training_data()
            return True
        except Exception as e:
            print(f"Error adding training data: {e}")
            return False
    
    def delete_training_data(self, index):
        """Delete training data by index"""
        try:
            if 0 <= index < len(self.training_data):
                del self.training_data[index]
                self.save_training_data()
                return True
            return False
        except Exception as e:
            print(f"Error deleting training data: {e}")
            return False
    
    def get_training_data(self):
        """Get all training data"""
        return self.training_data
    
    def train_model(self, training_data=None):
        """Train the chatbot model"""
        try:
            if training_data is None:
                training_data = self.training_data
            
            if not training_data:
                return {'trained_samples': 0, 'message': 'No training data available'}
            
            # Extract questions and answers
            questions = [item['question'] for item in training_data]
            answers = [item['answer'] for item in training_data]
            
            # Fit vectorizer and transform questions
            X = self.vectorizer.fit_transform(questions)
            
            # Fit label encoder and transform answers
            y = self.label_encoder.fit_transform(answers)
            
            # Train the model
            self.model = MultinomialNB()
            self.model.fit(X, y)
            
            # Save the model
            self.save_model()
            
            return {'trained_samples': len(training_data), 'message': 'Training completed successfully'}
        
        except Exception as e:
            print(f"Error training model: {e}")
            return {'trained_samples': 0, 'message': f'Training failed: {str(e)}'}
    
    def retrain_model(self):
        """Retrain the model with all current training data"""
        return self.train_model()
    
    def get_response(self, user_input):
        """Get response from the chatbot"""
        try:
            if self.model is None or not self.training_data:
                return "I haven't been trained yet. Please add some training data and train me first!"
            
            # Transform user input
            user_vector = self.vectorizer.transform([user_input])
            
            # Get prediction
            prediction = self.model.predict(user_vector)
            
            # Get the predicted answer
            predicted_answer = self.label_encoder.inverse_transform(prediction)[0]
            
            # Calculate confidence score
            probabilities = self.model.predict_proba(user_vector)
            confidence = np.max(probabilities)
            
            # If confidence is low, try similarity-based matching
            if confidence < 0.3:
                return self._similarity_based_response(user_input)
            
            return predicted_answer
        
        except Exception as e:
            print(f"Error getting response: {e}")
            return "I'm sorry, I encountered an error processing your message. Please try again."
    
    def _similarity_based_response(self, user_input):
        """Fallback response using similarity matching"""
        try:
            if not self.training_data:
                return "I don't have enough information to respond. Please train me with some data first!"
            
            # Calculate similarity with all training questions
            similarities = []
            user_input_lower = user_input.lower().strip()
            
            for item in self.training_data:
                question_lower = item['question'].lower().strip()
                
                # Exact match check first
                if user_input_lower == question_lower:
                    return item['answer']
                
                # Word overlap similarity
                user_words = set(user_input_lower.split())
                question_words = set(question_lower.split())
                
                if len(user_words) == 0 or len(question_words) == 0:
                    similarity = 0
                else:
                    intersection = len(user_words.intersection(question_words))
                    union = len(user_words.union(question_words))
                    similarity = intersection / union if union > 0 else 0
                
                similarities.append(similarity)
            
            # Get the most similar question
            max_similarity_idx = np.argmax(similarities)
            max_similarity = similarities[max_similarity_idx]
            
            if max_similarity > 0.2:  # Increased threshold for better matching
                return self.training_data[max_similarity_idx]['answer']
            else:
                return "I'm not sure how to respond to that. Could you please rephrase your question or add some training data to help me learn?"
        
        except Exception as e:
            print(f"Error in similarity-based response: {e}")
            return "I'm sorry, I'm having trouble understanding your question. Please try again."
