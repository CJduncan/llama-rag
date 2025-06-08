from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import chromadb
import openai
import os
import hashlib
from typing import List, Dict
import PyPDF2
import docx
from pathlib import Path
import time
from werkzeug.utils import secure_filename
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
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("    Using system environment variables instead.")
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
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    UPLOAD_FOLDER='uploads',
    SESSION_COOKIE_SECURE=os.environ.get('FLASK_ENV') == 'production',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    N8N_WEBHOOK_URL=os.environ.get('N8N_WEBHOOK_URL'),
    OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY'),
    OPENAI_MODEL=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
)

# Configure CORS for production
allowed_origins = os.environ.get('CORS_ORIGINS', 'http://localhost:8000,https://yourdomain.com').split(',')
CORS(app, origins=allowed_origins)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
if not app.config['OPENAI_API_KEY']:
    app.logger.error("❌ OPENAI_API_KEY not set! Please check your .env file.")
    print("❌ CRITICAL: OpenAI API key is required!")
    print("   Add this to your .env file: OPENAI_API_KEY=sk-your-key-here")
else:
    openai.api_key = app.config['OPENAI_API_KEY']
    app.logger.info("✅ OpenAI API key configured successfully")

class DocumentProcessor:
    """Handles document processing and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF files"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            app.logger.error(f"Error reading PDF: {e}")
        return text
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX files"""
        text = ""
        try:
            doc = docx.Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            app.logger.error(f"Error reading DOCX: {e}")
        return text
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from TXT files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            app.logger.error(f"Error reading TXT: {e}")
            return ""

class RAGSystem:
    """Main RAG system using OpenAI and ChromaDB for Business Automation Intelligence"""
    
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path="./chromadb")
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.doc_processor = DocumentProcessor()
        
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
        Optivus provides comprehensive AI-powered business automation solutions that transform how companies operate. We don't just process documents - we automate entire business workflows and processes.

        CORE AUTOMATION CAPABILITIES:

        1. INTELLIGENT DOCUMENT PROCESSING
        - Extract data from any document type (PDFs, Word, Excel, images, handwritten forms)
        - Automatically categorize and route documents
        - Convert unstructured data into structured business insights
        - Process invoices, contracts, reports, and forms automatically

        2. EMAIL AUTOMATION & MANAGEMENT
        - Read and analyze incoming emails automatically
        - Generate and send personalized email responses
        - Route emails to appropriate departments/people
        - Extract action items and deadlines from email conversations
        - Automatically follow up on pending communications

        3. REPORT ANALYSIS & INSIGHTS
        - Analyze financial reports, sales data, and performance metrics
        - Generate automated summaries and insights
        - Create executive dashboards with real-time data
        - Identify trends, anomalies, and opportunities
        - Provide data-driven recommendations

        4. MEETING & CALENDAR AUTOMATION
        - Schedule meetings based on availability and preferences
        - Send meeting invitations and reminders automatically
        - Analyze meeting notes and extract action items
        - Follow up on meeting commitments and deadlines
        - Coordinate complex multi-stakeholder scheduling

        5. WORKFLOW AUTOMATION
        - Connect different business systems and tools
        - Automate repetitive tasks and processes
        - Create intelligent decision trees for complex workflows
        - Handle approvals, notifications, and escalations
        - Streamline operations from lead to customer success

        6. CUSTOMER SERVICE AUTOMATION
        - Provide 24/7 intelligent customer support
        - Analyze customer inquiries and route appropriately
        - Generate personalized responses based on customer history
        - Escalate complex issues to human agents when needed
        - Track and resolve customer issues automatically

        7. DATA INTEGRATION & SYNC
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
        - Gain real-time insights into business performance

        INDUSTRIES WE SERVE:
        - Professional Services (Legal, Accounting, Consulting)
        - Healthcare and Medical Practices
        - Real Estate and Property Management
        - Financial Services and Insurance
        - Manufacturing and Supply Chain
        - E-commerce and Retail
        - Education and Training
        - Non-profits and Government

        IMPLEMENTATION APPROACH:
        - Start with high-impact, low-risk automation opportunities
        - Integrate seamlessly with existing systems
        - Provide comprehensive training and support
        - Scale automation gradually based on success
        - Continuous optimization and improvement

        SECURITY & COMPLIANCE:
        - Enterprise-grade security and encryption
        - GDPR, HIPAA, and industry-specific compliance
        - Data privacy and protection guaranteed
        - Audit trails and monitoring
        - On-premise or cloud deployment options

        GETTING STARTED:
        We offer free consultation sessions to identify automation opportunities specific to your business. Our experts analyze your current processes and provide a customized automation roadmap.

        NOTE: Pricing information is provided only upon request during consultation sessions, as solutions are customized based on specific business needs and requirements.
        """
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        return [chunk for chunk in chunks if chunk]
    
    def add_document(self, file_path: str, file_name: str) -> bool:
        """Add a document to the vector database"""
        try:
            # Extract text based on file type
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                text = self.doc_processor.extract_text_from_pdf(file_path)
            elif file_extension == '.docx':
                text = self.doc_processor.extract_text_from_docx(file_path)
            elif file_extension == '.txt':
                text = self.doc_processor.extract_text_from_txt(file_path)
            else:
                return False
            
            if not text.strip():
                return False
            
            # Split into chunks
            chunks = self.chunk_text(text)
            
            # Generate embeddings and store in ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                doc_id = f"{file_name}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
                documents.append(chunk)
                metadatas.append({
                    "source": file_name,
                    "chunk_id": i,
                    "file_type": file_extension
                })
                ids.append(doc_id)
            
            # Add to ChromaDB (ChromaDB will handle embeddings automatically)
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            app.logger.info(f"Successfully added document: {file_name}")
            return True
            
        except Exception as e:
            app.logger.error(f"Error adding document: {e}")
            return False
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant document chunks"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    search_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                    })
            
            return search_results
            
        except Exception as e:
            app.logger.error(f"Error searching documents: {e}")
            return []
    
    def check_pricing_request(self, query: str) -> bool:
        """Check if the user is specifically asking about pricing"""
        pricing_keywords = [
            'price', 'pricing', 'cost', 'costs', 'how much', 'expensive', 
            'cheap', 'budget', 'fee', 'fees', 'rate', 'rates', 'quote',
            'quotation', 'estimate', 'investment', 'roi calculation'
        ]
        return any(keyword in query.lower() for keyword in pricing_keywords)
    
    def generate_response(self, query: str, context: str, model: str = None) -> str:
        """Generate response using OpenAI with context"""
        try:
            if not app.config['OPENAI_API_KEY']:
                return "❌ OpenAI API key not configured. Please check your .env file and restart the server."
            
            # Use the configured model or default
            model = model or app.config['OPENAI_MODEL']
            
            # Check if pricing is requested
            pricing_requested = self.check_pricing_request(query)
            
            # Enhanced context with business automation focus
            enhanced_context = f"""
            BUSINESS AUTOMATION CONTEXT:
            {self.business_automation_context}
            
            DOCUMENT CONTEXT:
            {context}
            
            PRICING POLICY:
            {'The user is asking about pricing. You may discuss pricing and provide general cost information.' if pricing_requested else 'Do not mention specific prices, costs, or ROI figures unless explicitly asked. Instead, emphasize value and benefits, and offer a consultation for pricing details.'}
            """
            
            prompt = f"""You are an AI assistant for Optivus, a comprehensive business automation platform. Answer the user's question directly and concisely.

RESPONSE GUIDELINES:
- Keep responses SHORT (2-3 sentences max)
- Be conversational and friendly, not salesy
- Give specific, practical examples
- Don't over-explain or use buzzwords
- Sound natural and helpful
{f'- The user asked about pricing, so you can discuss costs briefly' if pricing_requested else '- Do not mention pricing unless asked'}

Context: {enhanced_context}

Question: {query}

Give a brief, helpful answer:"""
            
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are an expert in business automation and AI implementation. Help users understand how comprehensive automation can transform their operations.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                max_tokens=150,  # Much shorter responses
                temperature=0.7,
                timeout=30
            )
            
            return response.choices[0].message.content.strip()
            
        except openai.RateLimitError:
            app.logger.error("OpenAI rate limit exceeded")
            return "I'm experiencing high demand right now. Please try again in a moment."
        
        except openai.AuthenticationError:
            app.logger.error("OpenAI authentication failed - check API key")
            return "There's an authentication issue with the AI service. Please check the API key configuration."
        
        except openai.APITimeoutError:
            app.logger.error("OpenAI request timed out")
            return "The AI service is taking longer than expected. Please try again."
        
        except Exception as e:
            app.logger.error(f"Error generating response: {e}")
            return f"I'm sorry, I'm having trouble processing your request right now. Please try again later."
    
    def generate_no_context_response(self, query: str, model: str = None) -> str:
        """Generate response using OpenAI without document context but with business automation knowledge"""
        try:
            if not app.config['OPENAI_API_KEY']:
                return "❌ OpenAI API key not configured. Please check your .env file and restart the server.\n\nWould you like a free consultation? Just share your email and we'll reach out!"
            
            model = model or app.config['OPENAI_MODEL']
            
            # Check if pricing is requested
            pricing_requested = self.check_pricing_request(query)
            
            system_prompt = f"""You are an AI assistant for Optivus, a business automation platform. Give SHORT, helpful answers (2-3 sentences max).

WHAT WE DO: Automate emails, reports, scheduling, workflows, and business processes.

RESPONSE STYLE:
- Brief and conversational
- Specific examples when helpful
- Not salesy or over-enthusiastic
- Natural and friendly tone

PRICING: {'Discuss costs briefly if asked.' if pricing_requested else 'No pricing unless specifically asked.'}

Always end with a simple consultation offer."""
            
            response = openai.chat.completions.create(
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
                max_tokens=120,  # Shorter responses
                temperature=0.7,
                timeout=30
            )
            
            return response.choices[0].message.content.strip()
            
        except openai.RateLimitError:
            app.logger.error("OpenAI rate limit exceeded")
            return "I'm experiencing high demand right now. Please try again in a moment.\n\nWould you like a free consultation? Just share your email and we'll reach out!"
        
        except openai.AuthenticationError:
            app.logger.error("OpenAI authentication failed")
            return "There's an authentication issue with the AI service.\n\nWould you like a free consultation? Just share your email and we'll reach out!"
        
        except Exception as e:
            app.logger.error(f"Error generating no-context response: {e}")
            return "I'm having trouble right now, but I'd love to help! Our platform can automate your emails, reports, scheduling, and entire business workflows.\n\nWould you like a free consultation? Just share your email and we'll reach out!"
    
    def get_document_count(self) -> int:
        """Get number of documents in the collection"""
        try:
            return self.collection.count()
        except:
            return 0
# Initialize RAG system
rag_system = RAGSystem()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def extract_email_from_message(message):
    """Extract email address from a message"""
    # Pattern to find email addresses in text
    email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    matches = re.findall(email_pattern, message)
    
    # Return the first valid email found
    for match in matches:
        if is_valid_email(match):
            return match
    return None

def send_to_n8n(email, context=""):
    """Send email to n8n webhook"""
    try:
        webhook_url = app.config.get('N8N_WEBHOOK_URL')
        
        # Debug logging
        app.logger.info(f"Attempting to send email to n8n: {email}")
        app.logger.info(f"Webhook URL: {webhook_url}")
        
        if not webhook_url:
            app.logger.warning("N8N_WEBHOOK_URL not configured")
            return False
        
        payload = {
            'email': email,
            'timestamp': datetime.now().isoformat(),
            'source': 'ai-document-chat',
            'context': context,
            'ip': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', ''),
            'message': 'New consultation request from AI chat'
        }
        
        app.logger.info(f"Sending payload: {payload}")
        
        response = requests.post(
            webhook_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=15
        )
        
        app.logger.info(f"N8N response status: {response.status_code}")
        app.logger.info(f"N8N response text: {response.text}")
        
        return response.status_code in [200, 201, 202]
        
    except requests.exceptions.Timeout:
        app.logger.error("N8N webhook timeout")
        return False
    except requests.exceptions.RequestException as e:
        app.logger.error(f"N8N webhook request failed: {e}")
        return False
    except Exception as e:
        app.logger.error(f"Unexpected error sending to n8n: {e}")
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
        
        # Debug: Log all incoming messages
        app.logger.info(f"Received message: '{query}'")
        
        # Check if the message contains an email
        extracted_email = extract_email_from_message(query)
        app.logger.info(f"Extracted email: {extracted_email}")
        
        if extracted_email:
            # Send email to n8n
            success = send_to_n8n(extracted_email, "Consultation request from AI chat")
            
            if success:
                response_text = "Thank you! We've received your email and will be in touch within 24 hours to schedule your free consultation. We're excited to discuss how AI document intelligence can transform your business!"
            else:
                response_text = "Thank you for your interest! There was a technical issue capturing your email, but you can reach us directly at contact@yourdomain.com for a consultation."
            
            return jsonify({
                'response': response_text,
                'sources': 0,
                'email_captured': success
            })
        
        # Search for relevant documents
        search_results = rag_system.search_documents(query)
        
        if search_results:
            # Combine context from search results
            context = "\n\n".join([result['content'] for result in search_results])
            
            # Generate response with context
            response = rag_system.generate_response(query, context, model)
            
            # Add consultation offer to every response
            response += "\n\nWould you like a free consultation? Just share your email and we'll reach out!"
            
            return jsonify({
                'response': response,
                'sources': len(search_results)
            })
        else:
            # Generate response without document context but with business knowledge
            response = rag_system.generate_no_context_response(query, model)
            
            return jsonify({
                'response': response,
                'sources': 0
            })
            
    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Sorry, there was an error processing your request.'}), 500

@app.route('/upload', methods=['POST'])
def upload_document():
    """Handle document upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Add to RAG system
            success = rag_system.add_document(file_path, filename)
            
            if success:
                return jsonify({
                    'message': f'Document {filename} uploaded and processed successfully',
                    'filename': filename
                })
            else:
                return jsonify({'error': 'Failed to process document'}), 500
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        app.logger.error(f"Error uploading document: {e}")
        return jsonify({'error': 'Failed to upload document'}), 500

@app.route('/test-email/<email>')
def test_email_validation(email):
    """Test email validation"""
    is_valid = is_valid_email(email)
    return jsonify({
        'email': email,
        'is_valid': is_valid,
        'pattern_used': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    })

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        doc_count = rag_system.get_document_count()
        webhook_url = app.config.get('N8N_WEBHOOK_URL')
        openai_configured = bool(app.config.get('OPENAI_API_KEY'))
        
        return jsonify({
            'status': 'healthy',
            'document_count': doc_count,
            'webhook_configured': bool(webhook_url),
            'openai_configured': openai_configured,
            'openai_model': app.config.get('OPENAI_MODEL'),
            'timestamp': int(time.time()),
            'environment': os.environ.get('FLASK_ENV', 'development')
        })
    except Exception as e:
        app.logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/test-webhook', methods=['POST'])
def test_webhook():
    """Test endpoint to verify n8n webhook"""
    try:
        test_email = "test@example.com"
        success = send_to_n8n(test_email, "Test webhook from health check")
        
        return jsonify({
            'webhook_test': 'success' if success else 'failed',
            'test_email': test_email,
            'webhook_url': app.config.get('N8N_WEBHOOK_URL', 'Not configured')
        })
    except Exception as e:
        return jsonify({
            'webhook_test': 'error',
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
        
        # Simple test call
        response = openai.chat.completions.create(
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

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Print startup banner
    print("\n" + "="*60)
    print("🚀 OPTIVUS AI CHAT SERVER")
    print("="*60)
    
    # Validate critical environment variables
    if not os.environ.get('OPENAI_API_KEY'):
        print("❌ CRITICAL ERROR: OPENAI_API_KEY not found!")
        print("   Please create a .env file with your OpenAI API key:")
        print("   OPENAI_API_KEY=sk-your-key-here")
        print("\n   Or export it directly:")
        print("   export OPENAI_API_KEY='sk-your-key-here'")
        exit(1)
    
    if not os.environ.get('SECRET_KEY'):
        print("⚠️  WARNING: SECRET_KEY not set, using random key (sessions won't persist)")
        print("   Add to .env: SECRET_KEY=" + secrets.token_hex(32))
    
    # Configuration summary
    print(f"\n📋 Configuration:")
    print(f"   OpenAI Model: {os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}")
    print(f"   Environment: {os.environ.get('FLASK_ENV', 'development')}")
    print(f"   Port: {os.environ.get('PORT', '8000')}")
    print(f"   N8N Webhook: {'✅ Configured' if os.environ.get('N8N_WEBHOOK_URL') else '❌ Not set'}")
    
    # Production vs Development settings
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    port = int(os.environ.get('PORT', 8000))
    
    print(f"\n🌐 Starting server...")
    print(f"   Main page: http://localhost:{port}/")
    print(f"   Health check: http://localhost:{port}/health")
    print(f"   Test OpenAI: http://localhost:{port}/test-openai")
    
    if debug_mode:
        print("   Mode: 🛠  Development (debug enabled)")
    else:
        print("   Mode: 🏭 Production")
    
    print("="*60 + "\n")
    
    app.run(
        debug=debug_mode,
        host='0.0.0.0',
        port=port
    )