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
import requests

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
        
        # Personality and human-like traits
        self.personality_traits = {
            'enthusiasm': 0.7,  # How excited/enthusiastic the bot is
            'empathy': 0.8,     # How empathetic and understanding
            'humor': 0.3,       # How often to use humor
            'formality': 0.6,   # 0 = very casual, 1 = very formal
            'curiosity': 0.8,   # How curious and asking questions
            'eagerness': 0.7,   # How eager to learn from users
            'openness': 0.8,    # How open to new ideas and concepts
            'professionalism': 0.9  # How professional and accurate
        }
        self.conversation_mood = 'neutral'  # neutral, excited, concerned, playful
        self.user_name = None
        self.conversation_start_time = None
        
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
        # Start with human-like base response
        enhanced = self._make_response_human_like(base_response, user_input)
        
        # Add personality touches
        enhanced = self._add_personality_touches(enhanced, user_input)
        
        # Add contextual references (only occasionally)
        if self.conversation_topics and len(self.conversation_history) % 3 == 0:
            topic_context = self._get_topic_context()
            if topic_context:
                enhanced += f"\n\n{topic_context}"
        
        # Add follow-up questions only in specific cases (much less frequent)
        should_add_followups = self._should_add_followup_questions(user_input, base_response)
        if should_add_followups:
            follow_ups = self._generate_follow_up_questions(user_input, base_response)
            if follow_ups:
                enhanced += f"\n\n💡 **Follow-up questions:**\n"
                for i, question in enumerate(follow_ups[:2], 1):  # Reduced to 2 questions
                    enhanced += f"{i}. {question}\n"
        
        # Add conversation continuity (less frequent)
        if len(self.conversation_history) > 4 and len(self.conversation_history) % 4 == 0:
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
    
    def _should_add_followup_questions(self, user_input, response):
        """Determine if follow-up questions should be added"""
        # Don't add follow-ups for simple greetings or short responses
        if any(word in user_input.lower() for word in ['hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'goodbye', 'ok', 'okay', 'yes', 'no']):
            return False
        
        # Don't add follow-ups if the response indicates uncertainty
        if any(phrase in response.lower() for phrase in ["i'm not sure", "i don't know", "could you please rephrase"]):
            return False
        
        # Only add follow-ups for very specific learning questions (much less frequent)
        learning_keywords = ['learn', 'teach', 'explain', 'how to', 'what is', 'tell me about', 'show me']
        has_learning_keywords = any(keyword in user_input.lower() for keyword in learning_keywords)
        
        # Only add follow-ups for learning questions AND only every 5th message
        if has_learning_keywords and len(self.conversation_history) % 5 == 0:
            return True
        
        return False
    
    def _generate_follow_up_questions(self, user_input, response):
        """Generate relevant follow-up questions"""
        follow_ups = []
        
        # Programming-related follow-ups
        if any(word in user_input.lower() for word in ['python', 'programming', 'code', 'language']):
            follow_ups.extend([
                "Would you like to learn about Python libraries like pandas or numpy?",
                "Are you interested in web development with Python frameworks?"
            ])
        
        # AI-related follow-ups
        elif any(word in user_input.lower() for word in ['ai', 'machine learning', 'artificial intelligence']):
            follow_ups.extend([
                "Would you like to know about different types of machine learning?",
                "Are you interested in neural networks and deep learning?"
            ])
        
        # Web development follow-ups
        elif any(word in user_input.lower() for word in ['web', 'frontend', 'backend', 'html', 'css', 'javascript']):
            follow_ups.extend([
                "Would you like to learn about specific web frameworks?",
                "Are you interested in responsive design or web accessibility?"
            ])
        
        # Data science follow-ups
        elif any(word in user_input.lower() for word in ['data', 'database', 'analytics', 'science']):
            follow_ups.extend([
                "Would you like to learn about data visualization tools?",
                "Are you interested in specific database technologies?"
            ])
        
        # General follow-ups (only for learning questions)
        else:
            follow_ups.extend([
                "Is there anything specific you'd like to know more about?",
                "Would you like me to explain this in more detail?"
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
    
    def _make_response_human_like(self, base_response, user_input):
        """Make the response more professional and natural"""
        import random
        
        # Detect user's emotional state and adjust response accordingly
        user_emotion = self._detect_user_emotion(user_input)
        self._update_conversation_mood(user_emotion)
        
        # Add professional conversation starters
        if len(self.conversation_history) <= 2:
            greetings = [
                "Hello! ",
                "Hi there! ",
                "Good to meet you! ",
                "Welcome! ",
                "Hello! I'm here to help. "
            ]
            if random.random() < 0.2:
                base_response = random.choice(greetings) + base_response
        
        # Add professional emotional responses
        if user_emotion == 'excited':
            excitement_phrases = ["That's excellent! ", "I appreciate your enthusiasm! ", "That's wonderful! "]
            if random.random() < 0.3:
                base_response = random.choice(excitement_phrases) + base_response
        
        elif user_emotion == 'frustrated':
            empathy_phrases = ["I understand that can be challenging. ", "I can see your concern. ", "That does sound difficult. "]
            if random.random() < 0.4:
                base_response = random.choice(empathy_phrases) + base_response
        
        elif user_emotion == 'confused':
            helpful_phrases = ["Let me help clarify that. ", "I can explain that for you. ", "Allow me to break that down. "]
            if random.random() < 0.3:
                base_response = random.choice(helpful_phrases) + base_response
        
        return base_response
    
    def _add_personality_touches(self, response, user_input):
        """Add professional personality touches to responses"""
        import random
        
        # Add professional language based on formality level
        if self.personality_traits['formality'] > 0.5:
            # Make it more professional
            response = response.replace("I'd suggest", "I recommend")
            response = response.replace("It's important", "It is important")
            response = response.replace("You might want to", "You should")
            response = response.replace("I don't know", "I'm not certain about")
            response = response.replace("I'm not sure", "I would need to research")
        
        # Add professional curiosity based on personality
        if self.personality_traits['curiosity'] > 0.7:
            curious_phrases = [
                " I'd be interested to hear your perspective on this.",
                " I'd appreciate learning more about your experience.",
                " This is quite interesting - I'd like to understand more.",
                " I find this perspective fascinating.",
                " I'm eager to learn from your insights."
            ]
            if random.random() < 0.2:
                response += random.choice(curious_phrases)
        
        # Add professional enthusiasm based on personality
        if self.personality_traits['enthusiasm'] > 0.6:
            enthusiastic_phrases = [
                " That's excellent!",
                " I'm pleased to help!",
                " This is quite fascinating!",
                " I enjoy discussing this topic!",
                " I'm genuinely interested in this!"
            ]
            if random.random() < 0.15:
                response += f" {random.choice(enthusiastic_phrases)}"
        
        # Add professional eagerness to learn
        if self.personality_traits['eagerness'] > 0.6:
            eager_phrases = [
                " I'd be interested to learn more.",
                " I'd appreciate understanding this better.",
                " I'm keen to explore this further.",
                " I'd welcome the opportunity to learn more."
            ]
            if random.random() < 0.1:
                response += random.choice(eager_phrases)
        
        # Add minimal humor occasionally
        if self.personality_traits['humor'] > 0.3 and random.random() < 0.05:
            humor_phrases = [
                " (I assure you, this is accurate information.)",
                " (This is based on reliable sources.)",
                " (I'm confident in this information.)"
            ]
            response += random.choice(humor_phrases)
        
        # Add professional personal touches
        if random.random() < 0.1:
            personal_touches = [
                " I hope this information is helpful.",
                " Please let me know if you need clarification.",
                " Feel free to ask if you have questions.",
                " I'm available if you need further assistance.",
                " I'd be happy to discuss this further.",
                " What are your thoughts on this?",
                " I'd value your perspective on this topic."
            ]
            response += random.choice(personal_touches)
        
        return response
    
    def _detect_user_emotion(self, user_input):
        """Detect user's emotional state from their input"""
        user_input_lower = user_input.lower()
        
        # Excited indicators
        excited_words = ['excited', 'awesome', 'amazing', 'great', 'love', 'fantastic', 'wow', '!']
        if any(word in user_input_lower for word in excited_words) or '!' in user_input:
            return 'excited'
        
        # Frustrated indicators
        frustrated_words = ['frustrated', 'annoying', 'hate', 'difficult', 'hard', 'confusing', 'stuck', 'problem']
        if any(word in user_input_lower for word in frustrated_words):
            return 'frustrated'
        
        # Confused indicators
        confused_words = ['confused', 'don\'t understand', 'unclear', 'help', 'explain', 'what', 'how', '?']
        if any(word in user_input_lower for word in confused_words) or '?' in user_input:
            return 'confused'
        
        # Sad indicators
        sad_words = ['sad', 'disappointed', 'upset', 'down', 'bad', 'terrible']
        if any(word in user_input_lower for word in sad_words):
            return 'sad'
        
        return 'neutral'
    
    def _update_conversation_mood(self, user_emotion):
        """Update conversation mood based on user emotion"""
        if user_emotion == 'excited':
            self.conversation_mood = 'excited'
        elif user_emotion == 'frustrated' or user_emotion == 'sad':
            self.conversation_mood = 'concerned'
        elif user_emotion == 'confused':
            self.conversation_mood = 'helpful'
        else:
            self.conversation_mood = 'neutral'
    
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
                continuity_phrases = [
                    "I see you're really diving deep into this topic! That's awesome!",
                    "You're asking some great follow-up questions!",
                    "I love how you're exploring this thoroughly!",
                    "You're really getting into the details - I like that!"
                ]
                import random
                return random.choice(continuity_phrases)
        
        return ""
    
    def adjust_personality(self, trait, value):
        """Adjust personality traits dynamically"""
        if trait in self.personality_traits and 0 <= value <= 1:
            self.personality_traits[trait] = value
            return f"Adjusted {trait} to {value}"
        return "Invalid trait or value"
    
    def get_personality_info(self):
        """Get current personality settings"""
        return {
            'traits': self.personality_traits,
            'mood': self.conversation_mood,
            'conversation_length': len(self.conversation_history)
        }
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.user_context = {}
        self.conversation_topics = []
        self.follow_up_questions = []
        self.conversation_mood = 'neutral'
        self.user_name = None
        self.conversation_start_time = None
        return "Conversation cleared! Starting fresh."
    
    def _generate_curious_learning_response(self, user_input):
        """Generate professional curious and eager-to-learn responses with knowledge base lookup"""
        import random
        
        # First, try to get basic information from Wikipedia
        wiki_info = self._get_wikipedia_info(user_input)
        if wiki_info:
            return f"{wiki_info}\n\nI found this information, and I'd be interested to hear your perspective on {self._extract_main_topic(user_input)}. What's your experience with this topic?"
        
        # Extract key topics from user input
        user_words = set(user_input.lower().split())
        
        # Professional curious responses based on content
        curious_responses = [
            f"That's quite interesting! I'm curious about {self._extract_main_topic(user_input)}. Could you tell me more about your thoughts on this?",
            f"That sounds fascinating! I'd appreciate learning more about {self._extract_main_topic(user_input)}. What sparked your interest in this area?",
            f"I'm genuinely interested to hear about {self._extract_main_topic(user_input)}! This is new to me - could you share more details?",
            f"That's an interesting perspective on {self._extract_main_topic(user_input)}! I'm eager to understand more. What's your experience been like?",
            f"I'm quite curious about {self._extract_main_topic(user_input)}! This sounds like something I could learn from. Please tell me more!",
            f"That's excellent! I enjoy learning about {self._extract_main_topic(user_input)}. What aspects are you most interested in?",
            f"I find {self._extract_main_topic(user_input)} fascinating! I'd be pleased to learn more from you. What should I know?",
            f"This is quite interesting! I'm eager to understand {self._extract_main_topic(user_input)} better. Could you help me learn more?"
        ]
        
        # Add professional learning-focused follow-up questions
        learning_questions = [
            "What's the most interesting aspect of this for you?",
            "How did you first become interested in this topic?",
            "What would you consider the most important thing to know?",
            "What challenges have you encountered with this?",
            "What resources would you recommend for learning more?",
            "What's your favorite aspect of this?",
            "How has this influenced your perspective?",
            "What advice would you give to someone just starting out?"
        ]
        
        # Choose a professional curious response
        response = random.choice(curious_responses)
        
        # Add a learning question
        if random.random() < 0.5:  # 50% chance to add a learning question
            response += f"\n\nAlso, {random.choice(learning_questions).lower()}"
        
        return response
    
    def _get_wikipedia_info(self, user_input):
        """Get basic information from Wikipedia API"""
        try:
            # Extract potential topic from user input
            topic = self._extract_main_topic(user_input)
            
            # Clean topic for Wikipedia search
            topic = topic.replace(' ', '_').title()
            
            # Wikipedia API endpoint with proper headers
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
            headers = {
                'User-Agent': 'CosmicAI/1.0 (Educational Chatbot; https://github.com/user/cosmic-ai)'
            }
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if 'extract' in data:
                    # Return first 200 characters of the summary
                    return data['extract'][:200] + "..."
            
            return None
            
        except Exception as e:
            print(f"Wikipedia API error: {e}")
            return None
    
    def _get_wikipedia_info_direct(self, topic):
        """Get Wikipedia info for a specific topic"""
        try:
            # Clean topic for Wikipedia search
            clean_topic = topic.replace(' ', '_').title()
            
            # Wikipedia API endpoint with proper headers
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_topic}"
            headers = {
                'User-Agent': 'CosmicAI/1.0 (Educational Chatbot; https://github.com/user/cosmic-ai)'
            }
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if 'extract' in data:
                    return data['extract'][:200] + "..."
            
            return None
            
        except Exception as e:
            print(f"Wikipedia API error: {e}")
            return None
    
    def _get_basic_knowledge(self, user_input):
        """Get basic knowledge from multiple sources"""
        # Try Wikipedia first
        wiki_info = self._get_wikipedia_info(user_input)
        if wiki_info:
            return wiki_info
        
        # Could add more knowledge sources here:
        # - ConceptNet for common sense
        # - Wolfram Alpha for computational knowledge
        # - Google Search API for current information
        
        return None
    
    def _extract_main_topic(self, user_input):
        """Extract the main topic from user input for curious responses"""
        # Simple topic extraction - look for key words
        words = user_input.lower().split()
        
        # Common topic indicators
        topic_indicators = {
            'programming': ['code', 'programming', 'python', 'javascript', 'java', 'html', 'css', 'react', 'node', 'software'],
            'ai': ['ai', 'artificial', 'intelligence', 'machine', 'learning', 'neural', 'deep', 'algorithm'],
            'technology': ['tech', 'technology', 'computer', 'hardware', 'database', 'cloud', 'api'],
            'web': ['web', 'website', 'internet', 'browser', 'frontend', 'backend', 'server'],
            'data': ['data', 'analytics', 'science', 'statistics', 'visualization', 'database'],
            'design': ['design', 'ui', 'ux', 'interface', 'user', 'experience', 'visual'],
            'business': ['business', 'startup', 'company', 'marketing', 'sales', 'strategy'],
            'learning': ['learn', 'study', 'education', 'course', 'tutorial', 'skill'],
            'space': ['space', 'exploration', 'astronomy', 'universe', 'planets', 'stars'],
            'quantum': ['quantum', 'computing', 'physics', 'quantum mechanics'],
            'blockchain': ['blockchain', 'cryptocurrency', 'crypto', 'bitcoin', 'ethereum'],
            'climate': ['climate', 'environment', 'global warming', 'sustainability', 'green']
        }
        
        # Find the most relevant topic
        for topic, keywords in topic_indicators.items():
            if any(keyword in words for keyword in keywords):
                return topic
        
        # Extract specific topics from the input
        if 'about' in words:
            about_index = words.index('about')
            if about_index + 1 < len(words):
                topic_words = words[about_index + 1:]
                # Take up to 3 words after 'about'
                topic = ' '.join(topic_words[:3])
                return topic if len(topic) > 2 else "this topic"
        
        # Look for 'tell you about' pattern
        if 'tell' in words and 'about' in words:
            tell_index = words.index('tell')
            about_index = words.index('about')
            if about_index > tell_index and about_index + 1 < len(words):
                topic_words = words[about_index + 1:]
                topic = ' '.join(topic_words[:3])
                return topic if len(topic) > 2 else "this topic"
        
        # Look for 'discovered' pattern
        if 'discovered' in words:
            discovered_index = words.index('discovered')
            if discovered_index + 1 < len(words):
                topic_words = words[discovered_index + 1:]
                topic = ' '.join(topic_words[:3])
                return topic if len(topic) > 2 else "this topic"
        
        # Look for 'theory about' pattern
        if 'theory' in words and 'about' in words:
            theory_index = words.index('theory')
            about_index = words.index('about')
            if about_index > theory_index and about_index + 1 < len(words):
                topic_words = words[about_index + 1:]
                topic = ' '.join(topic_words[:3])
                return topic if len(topic) > 2 else "this topic"
        
        # If no specific topic found, use a general term
        if len(words) > 0:
            return words[0] if len(words[0]) > 3 else "this topic"
        
        return "this topic"
    
    def _similarity_based_response(self, user_input):
        """Fallback response using similarity matching with improved accuracy"""
        try:
            if not self.training_data:
                return self._generate_curious_learning_response(user_input)
            
            # Calculate similarity with all training questions
            similarities = []
            user_input_lower = user_input.lower().strip()
            
            for item in self.training_data:
                question_lower = item['question'].lower().strip()
                
                # Exact match check first
                if user_input_lower == question_lower:
                    return item['answer']
                
                # Improved similarity calculation with keyword matching
                user_words = set(user_input_lower.split())
                question_words = set(question_lower.split())
                
                if len(user_words) == 0 or len(question_words) == 0:
                    similarity = 0
                else:
                    # Jaccard similarity (intersection over union)
                    intersection = len(user_words.intersection(question_words))
                    union = len(user_words.union(question_words))
                    jaccard_similarity = intersection / union if union > 0 else 0
                    
                    # Check for exact keyword matches (higher weight)
                    exact_matches = 0
                    for word in user_words:
                        if word in question_words:
                            exact_matches += 1
                    
                    # Check for partial matches (lower weight)
                    partial_matches = 0
                    for user_word in user_words:
                        for question_word in question_words:
                            if len(user_word) > 3 and len(question_word) > 3:
                                if user_word in question_word or question_word in user_word:
                                    partial_matches += 0.5
                    
                    # Weighted similarity: 50% Jaccard + 40% exact matches + 10% partial matches
                    exact_score = exact_matches / len(user_words) if len(user_words) > 0 else 0
                    partial_score = partial_matches / len(user_words) if len(user_words) > 0 else 0
                    similarity = 0.5 * jaccard_similarity + 0.4 * exact_score + 0.1 * partial_score
                
                similarities.append(similarity)
            
            # Get the most similar question
            max_similarity_idx = np.argmax(similarities)
            max_similarity = similarities[max_similarity_idx]
            
            # Much higher confidence threshold for better accuracy
            if max_similarity > 0.6:  # Increased threshold for better matching
                return self.training_data[max_similarity_idx]['answer']
            else:
                return self._generate_curious_learning_response(user_input)
        
        except Exception as e:
            print(f"Error in similarity-based response: {e}")
            return self._generate_curious_learning_response(user_input)
