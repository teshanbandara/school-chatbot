from flask import Flask, render_template, request, jsonify
import json
import os
from openai import OpenAI
import re
from difflib import SequenceMatcher
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configuration - Using GPT-4o-mini
MODEL_CONFIG = {
    "model": "gpt-4o-mini",
    "max_tokens": 300,
    "temperature": 0.7
}

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class SchoolChatbot:
    def __init__(self, knowledge_base_path='faq.json'):
        """Initialize the chatbot with knowledge base"""
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
    
    def load_knowledge_base(self, path):
        """Load FAQ knowledge base from JSON file"""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Knowledge base file {path} not found. Creating empty knowledge base.")
            return {"faqs": []}
        except json.JSONDecodeError:
            print(f"Error reading {path}. Using empty knowledge base.")
            return {"faqs": []}
    
    def similarity(self, a, b):
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def search_knowledge_base(self, question, threshold=0.6):
        """Search for answers in the knowledge base using similarity matching"""
        question_lower = question.lower()
        best_match = None
        best_similarity = 0
        
        for faq in self.knowledge_base.get("faqs", []):
            # Check similarity with the main question
            similarity_score = self.similarity(question_lower, faq["question"])
            
            # Also check against keywords if they exist
            if "keywords" in faq:
                for keyword in faq["keywords"]:
                    keyword_similarity = self.similarity(question_lower, keyword.lower())
                    if keyword_similarity > similarity_score:
                        similarity_score = keyword_similarity
            
            # Check if question contains any of the keywords
            if "keywords" in faq:
                for keyword in faq["keywords"]:
                    if keyword.lower() in question_lower:
                        similarity_score = max(similarity_score, 0.8)
            
            if similarity_score > best_similarity and similarity_score >= threshold:
                best_similarity = similarity_score
                best_match = faq
        
        return best_match["answer"] if best_match else None
    
    def get_ai_response(self, question):
        """Get response from OpenAI API using GPT-4o-mini"""
        try:
            # Check if API key is available
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("❌ No API key found in environment variables")
                return "I'm sorry, but I'm currently unable to access my AI capabilities. Please contact the school administration for assistance."
            
            print(f"✅ API key found, length: {len(api_key)}")
            
            # Create a school-themed system prompt
            system_prompt = """You are a helpful school chatbot assistant. You provide friendly, informative responses about school-related topics. 
            Keep your responses concise but helpful. If you don't know something specific about the school, politely say so and suggest contacting the school office.
            Always maintain a positive, educational tone appropriate for students, parents, and staff."""
            
            print("🔄 Attempting to contact OpenAI API...")
            print(f"🤖 Using model: {MODEL_CONFIG['model']}")
            
            response = client.chat.completions.create(
                model=MODEL_CONFIG['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=MODEL_CONFIG['max_tokens'],
                temperature=MODEL_CONFIG['temperature']
            )
            
            print("✅ OpenAI API call successful!")
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"❌ Detailed API Error: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            
            # More specific error messages
            if "quota" in str(e).lower():
                return "I'm currently at my usage limit. Please try again later or contact the school office for immediate assistance."
            elif "api_key" in str(e).lower() or "authentication" in str(e).lower():
                return "I'm having authentication issues. Please contact the school administration for assistance."
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                return "I'm having network connectivity issues. Please check your internet connection and try again."
            elif "model" in str(e).lower():
                return f"The GPT-4o-mini model is not available. Please contact support. (Error: {type(e).__name__})"
            else:
                return f"I'm sorry, I'm having trouble processing your request right now (Error: {type(e).__name__}). Please try again later or contact the school office for immediate assistance."
    
    def get_response(self, question):
        """Main method to get chatbot response - checks knowledge base first, then AI"""
        # First, try to find answer in knowledge base
        kb_answer = self.search_knowledge_base(question)
        
        if kb_answer:
            return kb_answer
        
        # If no match in knowledge base, use AI
        return self.get_ai_response(question)

# Initialize chatbot
chatbot = SchoolChatbot()

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        user_question = data.get('question', '').strip()
        
        if not user_question:
            return jsonify({'error': 'Please enter a question'}), 400
        
        # Get response from chatbot
        response = chatbot.get_response(user_question)
        
        return jsonify({
            'question': user_question,
            'answer': response
        })
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Sorry, something went wrong. Please try again.'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY environment variable not set. AI features will be limited.")
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)