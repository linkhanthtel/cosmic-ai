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
        
        # Conversation memory and context
        self.conversation_history = []
        self.user_context = {}
        self.conversation_topics = []
        self.follow_up_questions = []
        
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
        """Get response from the chatbot with conversation context"""
        try:
            if self.model is None or not self.training_data:
                return "I haven't been trained yet. Please add some training data and train me first!"
            
            # Add to conversation history
            self.conversation_history.append({
                'user': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update conversation context
            self._update_conversation_context(user_input)
            
            # Get base response
            base_response = self._get_base_response(user_input)
            
            # Enhance response with context and follow-ups
            enhanced_response = self._enhance_response(base_response, user_input)
            
            # Add bot response to history
            self.conversation_history.append({
                'bot': enhanced_response,
                'timestamp': datetime.now().isoformat()
            })
            
            return enhanced_response
        
        except Exception as e:
            print(f"Error getting response: {e}")
            return "I'm sorry, I encountered an error processing your message. Please try again."
    
    def _get_base_response(self, user_input):
        """Get the base response from the model"""
        try:
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
            print(f"Error getting base response: {e}")
            return "I'm not sure how to respond to that. Could you please rephrase your question?"
    
    def _update_conversation_context(self, user_input):
        """Update conversation context and topics"""
        # Extract topics from user input
        words = user_input.lower().split()
        topic_keywords = {
            'programming': ['code', 'programming', 'python', 'javascript', 'java', 'html', 'css', 'react', 'node'],
            'ai': ['ai', 'artificial', 'intelligence', 'machine', 'learning', 'neural', 'deep'],
            'technology': ['tech', 'technology', 'computer', 'software', 'hardware', 'database'],
            'web': ['web', 'website', 'internet', 'browser', 'frontend', 'backend'],
            'data': ['data', 'database', 'sql', 'analytics', 'science']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in words for keyword in keywords):
                if topic not in self.conversation_topics:
                    self.conversation_topics.append(topic)
        
        # Keep only last 5 topics
        self.conversation_topics = self.conversation_topics[-5:]
    
    def _enhance_response(self, base_response, user_input):
        """Enhance response with context, follow-ups, and personality"""
        enhanced = base_response
        
        # Add contextual references
        if self.conversation_topics:
            topic_context = self._get_topic_context()
            if topic_context:
                enhanced += f"\n\n{topic_context}"
        
        # Add follow-up questions
        follow_ups = self._generate_follow_up_questions(user_input, base_response)
        if follow_ups:
            enhanced += f"\n\n💡 **Follow-up questions:**\n"
            for i, question in enumerate(follow_ups[:3], 1):
                enhanced += f"{i}. {question}\n"
        
        # Add conversation continuity
        if len(self.conversation_history) > 2:
            continuity = self._add_conversation_continuity()
            if continuity:
                enhanced += f"\n\n{continuity}"
        
        return enhanced
    
    def _get_topic_context(self):
        """Get context based on conversation topics"""
        if not self.conversation_topics:
            return ""
        
        recent_topics = self.conversation_topics[-2:]  # Last 2 topics
        
        context_responses = {
            'programming': "I notice you're interested in programming! I can help with various programming languages and concepts.",
            'ai': "AI and machine learning are fascinating topics! I'd be happy to discuss these further.",
            'technology': "Technology is a broad and exciting field! What specific aspect interests you most?",
            'web': "Web development is a great skill to have! I can help with both frontend and backend concepts.",
            'data': "Data science and analytics are crucial in today's world! I can help explain various data concepts."
        }
        
        return context_responses.get(recent_topics[-1], "")
    
    def _generate_follow_up_questions(self, user_input, response):
        """Generate relevant follow-up questions"""
        follow_ups = []
        
        # Programming-related follow-ups
        if any(word in user_input.lower() for word in ['python', 'programming', 'code', 'language']):
            follow_ups.extend([
                "Would you like to learn about Python libraries like pandas or numpy?",
                "Are you interested in web development with Python frameworks?",
                "Do you want to know about Python best practices?"
            ])
        
        # AI-related follow-ups
        elif any(word in user_input.lower() for word in ['ai', 'machine learning', 'artificial intelligence']):
            follow_ups.extend([
                "Would you like to know about different types of machine learning?",
                "Are you interested in neural networks and deep learning?",
                "Do you want to learn about AI applications in real-world scenarios?"
            ])
        
        # General follow-ups
        else:
            follow_ups.extend([
                "Is there anything specific you'd like to know more about?",
                "Would you like me to explain this in more detail?",
                "Do you have any related questions?"
            ])
        
        return follow_ups
    
    def _add_conversation_continuity(self):
        """Add conversation continuity based on history"""
        if len(self.conversation_history) < 4:
            return ""
        
        # Check if user is asking similar questions
        recent_questions = [entry['user'] for entry in self.conversation_history[-4:] if 'user' in entry]
        
        if len(recent_questions) >= 2:
            # Simple similarity check
            last_question = recent_questions[-1].lower()
            second_last_question = recent_questions[-2].lower()
            
            common_words = set(last_question.split()) & set(second_last_question.split())
            if len(common_words) >= 2:
                return "I see you're exploring this topic further! Feel free to ask any related questions."
        
        return ""
    
    def get_conversation_summary(self):
        """Get a summary of the current conversation"""
        if not self.conversation_history:
            return "No conversation yet."
        
        topics = ", ".join(self.conversation_topics) if self.conversation_topics else "General discussion"
        message_count = len([entry for entry in self.conversation_history if 'user' in entry])
        
        return f"Conversation topics: {topics} | Messages exchanged: {message_count}"
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.user_context = {}
        self.conversation_topics = []
        self.follow_up_questions = []
        return "Conversation cleared! Starting fresh."
    
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
