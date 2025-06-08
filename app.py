from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import re
import requests
from datetime import datetime
import secrets
import logging

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
CORS(app, origins=allowed_origins)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global OpenAI client variable
openai_client = None

def initialize_openai():
    """Initialize OpenAI client safely"""
    global openai_client
    try:
        if app.config['OPENAI_API_KEY']:
            openai_client = OpenAI(api_key=app.config['OPENAI_API_KEY'])
            app.logger.info("✅ OpenAI API key configured successfully")
            return True
        else:
            app.logger.error("❌ OPENAI_API_KEY not set!")
            return False
    except Exception as e:
        app.logger.error(f"❌ Failed to initialize OpenAI: {e}")
        return False

class SimpleChatSystem:
    """Simplified chat system without document storage"""
    
    def __init__(self):
        # Business automation knowledge base
        self.business_automation_context = """
        OPTIVUS BUSINESS AUTOMATION PLATFORM
        Key messaging:
- We are a SMALL BUSINESS helping other SMALL BUSINESSES succeed
- We understand the daily struggles of small business owners: wearing multiple hats, manual processes, never enough time
- Our automation is AFFORDABLE and designed specifically for small businesses (not enterprise)
- We focus on practical, immediate time-saving solutions

Common small business automation opportunities:
1. Email management (customer inquiries, follow-ups)
2. Appointment scheduling and reminders  
3. Invoice generation and tracking
4. Inventory alerts and management
5. Customer follow-up sequences
6. Lead capture and qualification
7. Social media posting and responses
8. Report generation from sales data
9. After-hours customer support
10. Integration between POS, accounting, and other tools

Perfect for: Local restaurants, salons, auto shops, real estate agents, retailers, contractors, medical practices, fitness studios, accounting firms, cleaning services.

Tone: Friendly, understanding, fellow small business owner. Use "we understand" and "as fellow business owners" language. Focus on ROI, time savings, and simplicity.

When they show interest or ask about getting started, guide them toward scheduling a free 15-minute consultation to discuss their specific business needs.

Never oversell - be helpful and genuine. If they're not ready, that's fine. Provide value regardless.

        WHAT WE DO:
        Optivus provides comprehensive AI-powered business automation solutions that transform how companies operate. We automate entire business workflows and processes.

        CORE AUTOMATION CAPABILITIES:

        1. EMAIL AUTOMATION & MANAGEMENT
        - Read and analyze incoming emails automatically
        - Generate and send personalized email responses
        - Route emails to appropriate departments/people
        - Extract action items and deadlines from email conversations
        - Automatically follow up on pending communications

        2. WORKFLOW AUTOMATION
        - Connect different business systems and tools
        - Automate repetitive tasks and processes
        - Create intelligent decision trees for complex workflows
        - Handle approvals, notifications, and escalations
        - Streamline operations from lead to customer success

        3. CUSTOMER SERVICE AUTOMATION
        - Provide 24/7 intelligent customer support
        - Analyze customer inquiries and route appropriately
        - Generate personalized responses based on customer history
        - Escalate complex issues to human agents when needed
        - Track and resolve customer issues automatically

        4. SCHEDULING & CALENDAR AUTOMATION
        - Schedule meetings based on availability and preferences
        - Send meeting invitations and reminders automatically
        - Follow up on meeting commitments and deadlines
        - Coordinate complex multi-stakeholder scheduling

        5. DATA INTEGRATION & SYNC
        - Connect CRMs, ERPs, and other business systems
        - Synchronize data across multiple platforms
        - Maintain data consistency and accuracy
        - Create unified views of business information
        - Automate data entry and updates

        BUSINESS BENEFITS:
        - Reduce manual work by 70-90%
        - Eliminate human errors in repetitive tasks
        - Operate 24/7 without breaks or holidays
        - Scale operations without hiring more staff
        - Free up employees for strategic, creative work
        - Improve customer response times dramatically
        - Ensure consistent quality and compliance

        GETTING STARTED:
        We offer free consultation sessions to identify automation opportunities specific to your business.
        """
    
    def check_pricing_request(self, query: str) -> bool:
        """Check if the user is specifically asking about pricing"""
        pricing_keywords = [
            'price', 'pricing', 'cost', 'costs', 'how much', 'expensive', 
            'cheap', 'budget', 'fee', 'fees', 'rate', 'rates', 'quote',
            'quotation', 'estimate', 'investment', 'roi calculation'
        ]
        return any(keyword in query.lower() for keyword in pricing_keywords)
    
    def generate_response(self, query: str, model: str = None) -> str:
        """Generate response using OpenAI with business automation knowledge"""
        try:
            # Initialize OpenAI if not already done
            if not openai_client:
                if not initialize_openai():
                    return "❌ OpenAI service not available. Please try again later.\n\nWould you like a free consultation? Just share your email and we'll reach out!"
            
            model = model or app.config['OPENAI_MODEL']
            
            # Check if pricing is requested
            pricing_requested = self.check_pricing_request(query)
            
            system_prompt = f"""You are an AI assistant for Optivus, a business automation platform. Give SHORT, helpful answers (2-3 sentences max).

WHAT WE DO: Automate emails, reports, scheduling, workflows, and business processes for small businesses.

RESPONSE STYLE:
- Brief and conversational
- Specific examples when helpful
- Not salesy or over-enthusiastic
- Natural and friendly tone

PRICING: {'Discuss costs briefly if asked.' if pricing_requested else 'No pricing unless specifically asked.'}

Always end with a simple consultation offer."""
            
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt
                    },
                    {
                        'role': 'user',
                        'content': f"Business automation context: {self.business_automation_context}\n\nUser question: {query}"
                    }
                ],
                max_tokens=120,
                temperature=0.7,
                timeout=30
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            app.logger.error(f"Error generating response: {e}")
            return "I'm having trouble right now, but I'd love to help! Our platform can automate your emails, reports, scheduling, and entire business workflows.\n\nWould you like a free consultation? Just share your email and we'll reach out!"

# Initialize chat system
chat_system = SimpleChatSystem()

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

def send_to_n8n(email, context=""):
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
        
        app.logger.info(f"Received message: '{query}'")
        
        # Check if the message contains an email
        extracted_email = extract_email_from_message(query)
        
        if extracted_email:
            # Send email to n8n
            success = send_to_n8n(extracted_email, "Consultation request from AI chat")
            
            if success:
                response_text = "Thank you! We've received your email and will be in touch within 24 hours to schedule your free consultation. We're excited to discuss how business automation can transform your operations!"
            else:
                response_text = "Thank you for your interest! There was a technical issue capturing your email, but you can reach us directly for a consultation."
            
            return jsonify({
                'response': response_text,
                'sources': 0,
                'email_captured': success
            })
        
        # Generate response
        response = chat_system.generate_response(query, model)
        
        # Add consultation offer to every response
        if "consultation" not in response.lower():
            response += "\n\nWould you like a free consultation? Just share your email and we'll reach out!"
        
        return jsonify({
            'response': response,
            'sources': 0
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
        # Initialize OpenAI if not already done
        if not openai_client:
            if not initialize_openai():
                return jsonify({
                    'openai_test': 'failed',
                    'error': 'API key not configured or initialization failed'
                }), 400
        
        response = openai_client.chat.completions.create(
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
    print("🚀 OPTIVUS AI CHAT SERVER (Simplified)")
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