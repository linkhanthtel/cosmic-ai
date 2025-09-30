#!/usr/bin/env python3
"""
Backend Model Training Script
Train the chatbot model from the backend without UI interaction
"""

import json
import os
import sys
from datetime import datetime
from chatbot import ChatBot

def load_training_data_from_file(filename):
    """Load training data from a JSON file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading training data from {filename}: {e}")
        return []

def add_training_data_batch(questions_answers):
    """Add multiple Q&A pairs at once"""
    bot = ChatBot()
    
    added_count = 0
    for question, answer in questions_answers:
        success = bot.add_training_data(question, answer)
        if success:
            added_count += 1
        else:
            print(f"Failed to add: Q: {question}")
    
    print(f"Added {added_count} training samples")
    return added_count

def train_from_file(filename):
    """Train the model with data from a file"""
    print(f"Loading training data from {filename}...")
    training_data = load_training_data_from_file(filename)
    
    if not training_data:
        print("No training data found!")
        return False
    
    print(f"Found {len(training_data)} training samples")
    
    # Initialize chatbot and train
    bot = ChatBot()
    bot.training_data = training_data
    bot.save_training_data()
    
    print("Training model...")
    result = bot.train_model()
    
    if result['trained_samples'] > 0:
        print(f"✅ Training successful! Trained {result['trained_samples']} samples")
        return True
    else:
        print(f"❌ Training failed: {result['message']}")
        return False

def add_and_train(questions_answers):
    """Add new data and retrain the model"""
    print("Adding new training data...")
    added_count = add_training_data_batch(questions_answers)
    
    if added_count > 0:
        print("Retraining model...")
        bot = ChatBot()
        result = bot.retrain_model()
        
        if result['trained_samples'] > 0:
            print(f"✅ Model retrained successfully with {result['trained_samples']} samples")
            return True
        else:
            print(f"❌ Retraining failed: {result['message']}")
            return False
    else:
        print("No new data added, skipping training")
        return False

def show_current_status():
    """Show current training data status"""
    bot = ChatBot()
    training_data = bot.get_training_data()
    
    print(f"\n📊 Current Training Data Status:")
    print(f"Total samples: {len(training_data)}")
    
    if training_data:
        print(f"Sample questions:")
        for i, item in enumerate(training_data[:5], 1):
            print(f"  {i}. {item['question']}")
        if len(training_data) > 5:
            print(f"  ... and {len(training_data) - 5} more")
    
    # Check if model exists
    model_exists = os.path.exists('models/chatbot_model.pkl')
    print(f"Model trained: {'✅ Yes' if model_exists else '❌ No'}")
    
    return len(training_data), model_exists

def interactive_training():
    """Interactive mode for adding training data"""
    print("🤖 Interactive Training Mode")
    print("=" * 40)
    print("Enter 'quit' to exit")
    print("Enter 'status' to see current status")
    print("Enter 'train' to retrain with current data")
    print()
    
    while True:
        question = input("Question: ").strip()
        
        if question.lower() == 'quit':
            break
        elif question.lower() == 'status':
            show_current_status()
            continue
        elif question.lower() == 'train':
            bot = ChatBot()
            result = bot.retrain_model()
            print(f"Training result: {result}")
            continue
        elif not question:
            continue
        
        answer = input("Answer: ").strip()
        if not answer:
            print("❌ Answer cannot be empty")
            continue
        
        # Add and train
        success = add_and_train([(question, answer)])
        if success:
            print("✅ Added and trained successfully!")
        else:
            print("❌ Failed to add or train")
        print()

def main():
    """Main function"""
    print("🤖 AI Chatbot Backend Training")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'status':
            show_current_status()
        
        elif command == 'train':
            # Retrain with current data
            bot = ChatBot()
            result = bot.retrain_model()
            print(f"Training result: {result}")
        
        elif command == 'file' and len(sys.argv) > 2:
            # Train from file
            filename = sys.argv[2]
            train_from_file(filename)
        
        elif command == 'add' and len(sys.argv) > 3:
            # Add single Q&A pair
            question = sys.argv[2]
            answer = sys.argv[3]
            success = add_and_train([(question, answer)])
            print("✅ Success!" if success else "❌ Failed!")
        
        elif command == 'interactive':
            interactive_training()
        
        else:
            print("Usage:")
            print("  python train_model.py status                    # Show current status")
            print("  python train_model.py train                     # Retrain with current data")
            print("  python train_model.py file <filename>           # Train from JSON file")
            print("  python train_model.py add <question> <answer>   # Add single Q&A pair")
            print("  python train_model.py interactive               # Interactive mode")
    
    else:
        # Show menu
        print("Choose an option:")
        print("1. Show current status")
        print("2. Retrain with current data")
        print("3. Interactive training mode")
        print("4. Train from file")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            show_current_status()
        elif choice == '2':
            bot = ChatBot()
            result = bot.retrain_model()
            print(f"Training result: {result}")
        elif choice == '3':
            interactive_training()
        elif choice == '4':
            filename = input("Enter filename: ").strip()
            if filename:
                train_from_file(filename)
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
