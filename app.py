from flask import Flask, request, jsonify, render_template
import requests
import re
from datetime import datetime
import os

app = Flask(__name__)

# Configuration - Update this with your actual n8n webhook URL
N8N_WEBHOOK_URL = "https://cdmcatx69.app.n8n.cloud/webhook/3e2f713d-441c-462d-a206-5ced7dc503e4"
OLLAMA_URL = "http://localhost:11434/api/generate"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/more')
def more():
    return render_template('more.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        message_lower = message.lower()
        awaiting_scheduling_decision = data.get('awaiting_scheduling_decision', False)
        awaiting_email_and_time = data.get('awaiting_email_and_time', False)
        
        print(f"Received message: {message}")
        print(f"Awaiting scheduling decision: {awaiting_scheduling_decision}")
        print(f"Awaiting email and time: {awaiting_email_and_time}")
        
        # Step 1: Check if we're waiting for scheduling decision (yes/no)
        if awaiting_scheduling_decision:
            if any(word in message_lower for word in ['yes', 'sure', 'okay', 'ok', 'schedule', 'book', 'definitely', 'absolutely']):
                return jsonify({
                    'response': "Great! Please provide your email address and preferred time in one message. For example: 'john@email.com tomorrow at 2pm' or 'sarah@company.com next Tuesday morning'",
                    'ask_for_email_and_time': True
                })
            else:
                return jsonify({
                    'response': "No problem! Feel free to ask me any other questions about how AI document intelligence can help your business.",
                    'conversation_reset': True
                })
        
        # Step 2: Check if we're waiting for email and time
        elif awaiting_email_and_time:
            email_and_time = parse_email_and_time(message)
            
            if email_and_time and email_and_time['email'] and email_and_time['time']:
                print(f"Parsed email: {email_and_time['email']}, time: {email_and_time['time']}")
                
                # Send to n8n for appointment processing
                appointment_result = send_to_n8n_appointment(
                    email_and_time['email'], 
                    email_and_time['time']
                )
                
                print(f"n8n appointment result: {appointment_result}")
                
                if appointment_result.get('success'):
                    return jsonify({
                        'response': appointment_result.get('message', 'Perfect! I\'ve scheduled your consultation. You\'ll receive a confirmation email shortly with all the details.'),
                        'appointment_confirmed': True
                    })
                elif appointment_result.get('busy'):
                    return jsonify({
                        'response': appointment_result.get('message', 'That time slot is already booked. Could you suggest another time?'),
                        'appointment_busy': True
                    })
                else:
                    return jsonify({
                        'response': "I'm having trouble with scheduling right now. Could you try again in a moment, or I'll have someone reach out to you directly?",
                        'appointment_busy': True
                    })
            else:
                return jsonify({
                    'response': "I couldn't find both an email and time in your message. Please try again with format like: 'john@email.com tomorrow at 2pm'",
                    'ask_for_email_and_time': True
                })
        
        # Step 3: Check if this is just an email (for original email capture flow)
        elif is_valid_email(message):
            print(f"Valid email detected: {message}")
            email_result = send_to_n8n_email(message)
            
            if email_result.get('success'):
                return jsonify({
                    'response': "Thank you! We've received your email and will be in touch within 24 hours to schedule your free consultation. We're excited to discuss how AI document intelligence can transform your business!",
                    'email_captured': True
                })
            else:
                return jsonify({
                    'response': "Thank you for your email! There was a small issue saving it, but I've noted it down. We'll be in touch soon!",
                    'email_captured': True
                })
        
        # Step 4: Regular conversation - check if they want to schedule
        else:
            # Check if message indicates wanting to schedule
            schedule_keywords = [
                'schedule', 'appointment', 'meeting', 'consultation', 'book', 
                'call', 'demo', 'talk', 'discuss', 'meet', 'setup', 'set up',
                'calendar', 'available', 'time', 'when can'
            ]
            wants_to_schedule = any(keyword in message_lower for keyword in schedule_keywords)
            
            if wants_to_schedule:
                return jsonify({
                    'response': "I'd be happy to help you schedule a consultation! Our AI document intelligence platform can transform how your business processes documents, extracts insights, and automates workflows. Would you like to schedule a free 30-minute consultation to discuss your specific needs?",
                    'ask_for_scheduling': True
                })
            else:
                # Regular AI chat using Ollama
                ai_response = get_ai_response(message)
                return jsonify({'response': ai_response})
    
    except Exception as e:
        print(f"Error in chat route: {str(e)}")
        return jsonify({'error': 'Something went wrong. Please try again.'}), 500

def is_valid_email(email):
    """Check if string is a valid email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def parse_email_and_time(message):
    """Extract email and time from user message"""
    email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    email_match = re.search(email_pattern, message)
    
    if email_match:
        email = email_match.group(1)
        # Remove email from message to get time portion
        time_text = re.sub(email_pattern, '', message).strip()
        # Clean up the time text
        time_text = re.sub(r'^[,\s]+|[,\s]+$', '', time_text)  # Remove leading/trailing commas and spaces
        
        return {
            'email': email,
            'time': time_text if time_text else None
        }
    return None

def send_to_n8n_email(email):
    """Send email to n8n workflow for processing"""
    try:
        payload = {
            'type': 'email',
            'email': email,
            'timestamp': datetime.now().isoformat(),
            'source': 'website_chat'
        }
        
        print(f"Sending email to n8n: {payload}")
        
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            timeout=10
        )
        
        print(f"n8n email response status: {response.status_code}")
        
        if response.status_code == 200:
            return {'success': True}
        else:
            print(f"n8n email error: {response.text}")
            return {'success': False}
            
    except Exception as e:
        print(f"Error sending email to n8n: {str(e)}")
        return {'success': False}

def send_to_n8n_appointment(email, time_preference):
    """Send appointment request to n8n workflow"""
    try:
        payload = {
            'type': 'appointment',
            'email': email,
            'time_preference': time_preference,
            'timestamp': datetime.now().isoformat(),
            'source': 'website_chat'
        }
        
        print(f"Sending appointment to n8n: {payload}")
        
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            timeout=15
        )
        
        print(f"n8n appointment response status: {response.status_code}")
        print(f"n8n appointment response: {response.text}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"n8n appointment parsed result: {result}")
                return result
            except:
                # If response isn't JSON, assume success
                return {'success': True, 'message': 'Appointment request processed successfully.'}
        else:
            print(f"n8n appointment error: {response.text}")
            return {'success': False, 'message': 'Scheduling service unavailable right now.'}
            
    except Exception as e:
        print(f"Error sending appointment to n8n: {str(e)}")
        return {'success': False, 'message': 'Could not process appointment request.'}

def get_ai_response(message):
    """Get AI response from Ollama"""
    try:
        # Enhanced system prompt for business context
        system_prompt = """You are an AI assistant for Optivus, an AI document intelligence platform. 
        You help businesses understand how AI can transform their document processing, data extraction, and workflow automation.
        
        Key capabilities you can discuss:
        - Document parsing and data extraction
        - Automated workflow processing
        - Business intelligence from documents
        - Integration with existing systems
        - ROI and efficiency improvements
        
        Be helpful, professional, and focus on business value. If someone asks about scheduling or meetings, 
        suggest they can schedule a consultation."""
        
        full_prompt = f"{system_prompt}\n\nUser question: {message}\n\nResponse:"
        
        response = requests.post(
            OLLAMA_URL,
            json={
                'model': 'llama3.2',
                'prompt': full_prompt,
                'stream': False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            ai_response = response.json().get('response', '').strip()
            if ai_response:
                return ai_response
            else:
                return "I'd be happy to help you learn about AI document intelligence! What specific aspect interests you most?"
        else:
            print(f"Ollama error: {response.status_code} - {response.text}")
            return "I'm here to help you understand how AI can transform your business operations. What would you like to know?"
    
    except Exception as e:
        print(f"Error getting AI response: {str(e)}")
        return "I'm here to help you learn about AI solutions for your business. What questions do you have?"

if __name__ == '__main__':
    # Update this URL with your actual n8n webhook URL
    print("🚀 Starting Flask app...")
    print(f"📧 n8n webhook: {N8N_WEBHOOK_URL}")
    print("💡 Don't forget to update your n8n webhook URL!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)