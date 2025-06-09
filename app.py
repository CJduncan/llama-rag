from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import openai
import os
import re
import requests
from datetime import datetime
import secrets
import logging
from collections import defaultdict

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file successfully")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables.")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")

app = Flask(__name__)

# Debug: Print environment variable status
print(f"🔍 Environment Check:")
print(f"   OPENAI_API_KEY: {'✅ SET' if os.environ.get('OPENAI_API_KEY') else '❌ MISSING'}")
print(f"   SECRET_KEY: {'✅ SET' if os.environ.get('SECRET_KEY') else '❌ MISSING'}")
print(f"   N8N_WEBHOOK_URL: {'✅ SET' if os.environ.get('N8N_WEBHOOK_URL') else '❌ MISSING'}")

# Configure app settings
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', secrets.token_hex(32)),
    SESSION_COOKIE_SECURE=os.environ.get('FLASK_ENV') == 'production',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    N8N_WEBHOOK_URL=os.environ.get('N8N_WEBHOOK_URL'),
    OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY'),
    OPENAI_MODEL=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
)

# Configure CORS for production
allowed_origins = os.environ.get('CORS_ORIGINS', '*').split(',')
CORS(app, origins=allowed_origins, supports_credentials=True)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI with legacy approach
if app.config['OPENAI_API_KEY']:
    openai.api_key = app.config['OPENAI_API_KEY'].strip()
    app.logger.info("✅ OpenAI API key configured successfully")
else:
    app.logger.error("❌ OPENAI_API_KEY not set!")

# In-memory conversation storage (use Redis in production)
conversations = defaultdict(list)

class SimpleChatSystem:
    """Simplified chat system with conversation history"""
    
    def __init__(self):
        # Business automation knowledge base
        self.business_automation_context = """
        OPTIVUS BUSINESS AUTOMATION PLATFORM
        
        We are a SMALL BUSINESS helping other SMALL BUSINESSES succeed through affordable automation.
        
        WHAT WE DO: Automate emails, reports, scheduling, workflows, and business processes for small businesses.
        
        CORE SERVICES:
        1. Email automation & management
        2. Workflow automation  
        3. Customer service automation
        4. Scheduling & calendar automation
        5. Data integration & sync
        
        BENEFITS: Reduce manual work by 70-90%, eliminate errors, operate 24/7, scale without hiring.
        
        PERFECT FOR: Restaurants, salons, auto shops, real estate agents, retailers, contractors, medical practices, fitness studios.
        """
    
    def check_pricing_request(self, query: str) -> bool:
        """Check if the user is specifically asking about pricing"""
        pricing_keywords = [
            'price', 'pricing', 'cost', 'costs', 'how much', 'expensive', 
            'cheap', 'budget', 'fee', 'fees', 'rate', 'rates', 'quote',
            'quotation', 'estimate', 'investment', 'roi calculation'
        ]
        return any(keyword in query.lower() for keyword in pricing_keywords)
    
    def should_ask_for_email(self, query: str, conversation_history: list) -> bool:
        """Determine if we should ask for email based on conversation"""
        # Keywords that indicate interest
        interest_keywords = [
            'interested', 'how do i start', 'get started', 'sign up', 'learn more',
            'tell me more', 'sounds good', 'want to try', 'need help', 'can you help',
            'pricing', 'cost', 'consultation', 'schedule', 'meeting', 'call'
        ]
        
        # Check if user seems interested
        user_interested = any(keyword in query.lower() for keyword in interest_keywords)
        
        # Check if we haven't asked for email recently (last 3 messages)
        recent_messages = conversation_history[-6:] if len(conversation_history) >= 6 else conversation_history
        asked_for_email_recently = any('email' in msg.get('content', '').lower() and msg.get('role') == 'assistant' 
                                     for msg in recent_messages)
        
        # Ask for email if user shows interest and we haven't asked recently
        return user_interested and not asked_for_email_recently
    
    def generate_response(self, query: str, session_id: str, model: str = None) -> str:
        """Generate response using OpenAI with conversation history"""
        try:
            if not app.config['OPENAI_API_KEY']:
                return "❌ OpenAI service not available. Please try again later.\n\nWould you like a free consultation? Just share your email and we'll reach out!"
            
            model = model or app.config['OPENAI_MODEL']
            
            # Get conversation history
            conversation_history = conversations[session_id]
            
            # Check if pricing is requested
            pricing_requested = self.check_pricing_request(query)
            
            # Check if we should ask for email
            ask_for_email = self.should_ask_for_email(query, conversation_history)
            
            system_prompt = f"""You are an AI assistant for Optivus, a business automation platform for small businesses.

PERSONALITY: Friendly, helpful, understanding fellow small business owner.

WHAT WE DO: Automate emails, reports, scheduling, workflows, and business processes for small businesses like restaurants, salons, contractors, etc.

RESPONSE STYLE:
- Keep responses SHORT (2-3 sentences max)
- Be conversational and natural
- Show you understand small business struggles
- Give specific examples when helpful
- Don't be overly salesy

PRICING: {'Mention that we offer affordable automation solutions designed for small business budgets if asked.' if pricing_requested else 'No pricing unless specifically asked.'}

EMAIL CAPTURE: {'Ask for their email to schedule a free consultation after your response.' if ask_for_email else 'Only ask for email if they show clear interest in getting started.'}

Remember previous conversation context and build on it naturally."""
            
            # Build messages with conversation history
            messages = [{'role': 'system', 'content': system_prompt}]
            
            # Add conversation history (last 10 messages to keep context reasonable)
            recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
            messages.extend(recent_history)
            
            # Add current user message
            messages.append({'role': 'user', 'content': query})
            
            # Call OpenAI
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=150,
                temperature=0.7,
                timeout=30
            )
            
            assistant_response = response.choices[0].message.content.strip()
            
            # Add email ask if needed and not already included
            if ask_for_email and 'email' not in assistant_response.lower():
                assistant_response += "\n\nWould you like to schedule a free 15-minute consultation? Just share your email and I'll have someone reach out to discuss your specific automation needs!"
            
            # Store conversation history
            conversations[session_id].append({'role': 'user', 'content': query})
            conversations[session_id].append({'role': 'assistant', 'content': assistant_response})
            
            # Keep conversation history manageable (last 20 messages)
            if len(conversations[session_id]) > 20:
                conversations[session_id] = conversations[session_id][-20:]
            
            return assistant_response
            
        except Exception as e:
            app.logger.error(f"Error generating response: {e}")
            return "I'm having trouble right now, but I'd love to help! Our platform automates emails, reports, scheduling, and workflows for small businesses.\n\nWould you like a free consultation? Just share your email and we'll reach out!"

# Initialize chat system
chat_system = SimpleChatSystem()

def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(16)
    return session['session_id']

def is_valid_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def extract_email_from_message(message):
    """Extract email address from a message"""
    email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    matches = re.findall(email_pattern, message)
    
    for match in matches:
        if is_valid_email(match):
            return match
    return None

def send_to_n8n(email, context="", session_id=""):
    """Send email to n8n webhook"""
    try:
        webhook_url = app.config.get('N8N_WEBHOOK_URL')
        
        app.logger.info(f"Attempting to send email to n8n: {email}")
        
        if not webhook_url:
            app.logger.warning("N8N_WEBHOOK_URL not configured")
            return False
        
        payload = {
            'email': email,
            'timestamp': datetime.now().isoformat(),
            'source': 'ai-chat',
            'context': context,
            'session_id': session_id,
            'ip': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', ''),
            'message': 'New consultation request from AI chat'
        }
        
        response = requests.post(
            webhook_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=15
        )
        
        app.logger.info(f"N8N response status: {response.status_code}")
        return response.status_code in [200, 201, 202]
        
    except Exception as e:
        app.logger.error(f"Error sending to n8n: {e}")
        return False

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/more')
def more():
    """Serve the more information page"""
    return render_template('more.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        query = data.get('message', '').strip()
        model = data.get('model', app.config['OPENAI_MODEL'])
        
        if not query:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get session ID for conversation tracking
        session_id = get_session_id()
        
        app.logger.info(f"Received message: '{query}' (Session: {session_id})")
        
        # Check if the message contains an email
        extracted_email = extract_email_from_message(query)
        
        if extracted_email:
            # Send email to n8n
            success = send_to_n8n(extracted_email, "Consultation request from AI chat", session_id)
            
            # Store the email capture in conversation history
            conversations[session_id].append({'role': 'user', 'content': query})
            
            if success:
                response_text = "Perfect! Thank you for sharing your email. We'll be in touch within 24 hours to schedule your free consultation. I'm excited for you to see how business automation can transform your operations and save you hours every week!"
            else:
                response_text = "Thank you for your email! There was a technical issue on our end, but don't worry - you can also reach us directly at our website. We'd love to discuss how automation can help your business!"
            
            # Store response in conversation history
            conversations[session_id].append({'role': 'assistant', 'content': response_text})
            
            return jsonify({
                'response': response_text,
                'sources': 0,
                'email_captured': success,
                'session_id': session_id
            })
        
        # Generate response with conversation history
        response = chat_system.generate_response(query, session_id, model)
        
        return jsonify({
            'response': response,
            'sources': 0,
            'session_id': session_id
        })
            
    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Sorry, there was an error processing your request.'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        webhook_url = app.config.get('N8N_WEBHOOK_URL')
        openai_configured = bool(app.config.get('OPENAI_API_KEY'))
        
        return jsonify({
            'status': 'healthy',
            'webhook_configured': bool(webhook_url),
            'openai_configured': openai_configured,
            'openai_model': app.config.get('OPENAI_MODEL'),
            'active_conversations': len(conversations),
            'timestamp': datetime.now().isoformat(),
            'environment': os.environ.get('FLASK_ENV', 'development')
        })
    except Exception as e:
        app.logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/test-openai', methods=['GET'])
def test_openai():
    """Test endpoint to verify OpenAI API"""
    try:
        if not app.config['OPENAI_API_KEY']:
            return jsonify({
                'openai_test': 'failed',
                'error': 'API key not configured'
            }), 400
        
        # Use legacy OpenAI API call
        response = openai.ChatCompletion.create(
            model=app.config['OPENAI_MODEL'],
            messages=[{'role': 'user', 'content': 'Hello, respond with just "OK"'}],
            max_tokens=10,
            timeout=10
        )
        
        return jsonify({
            'openai_test': 'success',
            'model': app.config['OPENAI_MODEL'],
            'response': response.choices[0].message.content.strip()
        })
        
    except Exception as e:
        return jsonify({
            'openai_test': 'failed',
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Print startup banner
    print("\n" + "="*60)
    print("🚀 OPTIVUS AI CHAT SERVER (With History & Email)")
    print("="*60)
    
    # Validate critical environment variables
    if not os.environ.get('OPENAI_API_KEY'):
        print("❌ CRITICAL ERROR: OPENAI_API_KEY not found!")
        print("   Please set your OpenAI API key in Railway environment variables")
        exit(1)
    
    if not os.environ.get('SECRET_KEY'):
        print("⚠️  WARNING: SECRET_KEY not set, generating random key")
    
    # Configuration summary
    print(f"\n📋 Configuration:")
    print(f"   OpenAI Model: {os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}")
    print(f"   Environment: {os.environ.get('FLASK_ENV', 'development')}")
    print(f"   Port: {os.environ.get('PORT', '8000')}")
    print(f"   N8N Webhook: {'✅ Configured' if os.environ.get('N8N_WEBHOOK_URL') else '❌ Not set'}")
    
    # Railway uses PORT environment variable
    port = int(os.environ.get('PORT', 8000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"\n🌐 Starting server on port {port}...")
    print(f"   Health check: http://localhost:{port}/health")
    print(f"   Test OpenAI: http://localhost:{port}/test-openai")
    
    if debug_mode:
        print("   Mode: 🛠  Development (debug enabled)")
    else:
        print("   Mode: 🏭 Production")
    
    print("="*60 + "\n")
    
    try:
        app.run(
            debug=debug_mode,
            host='0.0.0.0',
            port=port,
            threaded=True
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        exit(1)