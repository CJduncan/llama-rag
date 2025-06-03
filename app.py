from flask import Flask, request, jsonify, render_template
import requests
import re
import json
from datetime import datetime
import os
import logging

app = Flask(__name__)

# Configuration - UPDATE THIS WITH YOUR ACTUAL WEBHOOK URL FROM N8N
N8N_AI_AGENT_WEBHOOK = "https://cdmcatx69.app.n8n.cloud/webhook/4a6fef26-13d2-4b2f-91ff-31db71d58063"  # Production webhook
OLLAMA_URL = "http://localhost:11434/api/generate"

# In-memory conversation storage
conversation_history = {}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        conversation_id = data.get('conversation_id', f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        awaiting_scheduling_response = data.get('awaiting_scheduling_response', False)
        
        logger.info(f"Received message: {message[:50]}...")
        logger.info(f"Awaiting scheduling response: {awaiting_scheduling_response}")
        
        # Initialize conversation history if new
        if conversation_id not in conversation_history:
            conversation_history[conversation_id] = {
                'messages': [],
                'transferred_to_agent': False
            }
        
        conv_data = conversation_history[conversation_id]
        
        # Add user message to history
        conv_data['messages'].append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # If we're waiting for scheduling response (yes/no)
        if awaiting_scheduling_response:
            if is_affirmative_response(message):
                logger.info("User wants to schedule, transferring to n8n AI agent")
                
                # Mark as transferred
                conv_data['transferred_to_agent'] = True
                
                # Create context summary for n8n
                conversation_summary = create_conversation_summary(conv_data)
                transfer_message = f"User wants to schedule a consultation. Previous conversation context: {conversation_summary}"
                
                # Send to n8n AI agent
                scheduling_response = send_to_n8n_ai_agent(transfer_message, conversation_id, conv_data)
                
                if scheduling_response.get('success'):
                    response_data = scheduling_response.get('data', {})
                    logger.info(f"N8N Response Data: {response_data}")
                    
                    # Handle both 'message' and 'output' keys from N8N
                    ai_response = (
                        response_data.get('message') or 
                        response_data.get('output') or 
                        'Great! Let me help you schedule that consultation. When would work best for you?'
                    )
                    
                    logger.info(f"Extracted AI Response: {ai_response}")
                    
                    conv_data['messages'].append({
                        'role': 'assistant',
                        'content': ai_response,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'n8n_agent'
                    })
                    
                    return jsonify({
                        'response': ai_response,
                        'conversation_id': conversation_id,
                        'transferred_to_agent': True,
                        'action_type': 'scheduling_initiated'
                    })
                else:
                    # Fallback if n8n unavailable
                    ai_response = "Perfect! I'd love to help you schedule a consultation. Please provide your email and preferred time (e.g., 'john@email.com tomorrow at 2pm') and I'll get that set up for you."
                    
                    conv_data['messages'].append({
                        'role': 'assistant',
                        'content': ai_response,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'manual_scheduling'
                    })
                    
                    return jsonify({
                        'response': ai_response,
                        'conversation_id': conversation_id,
                        'transferred_to_agent': False,
                        'action_type': 'manual_scheduling'
                    })
            else:
                # User declined scheduling
                ai_response = "No problem! Feel free to ask me any other questions about AI document intelligence and how it can help your business."
                
                conv_data['messages'].append({
                    'role': 'assistant',
                    'content': ai_response,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'local_llm'
                })
                
                return jsonify({
                    'response': ai_response,
                    'conversation_id': conversation_id,
                    'awaiting_scheduling_response': False
                })
        
        # If already transferred to n8n, route everything there
        elif conv_data['transferred_to_agent']:
            logger.info("Routing to n8n agent - user is in scheduling flow")
            
            scheduling_response = send_to_n8n_ai_agent(message, conversation_id, conv_data)
            
            if scheduling_response.get('success'):
                response_data = scheduling_response.get('data', {})
                
                # Handle both 'message' and 'output' keys from N8N
                ai_response = (
                    response_data.get('message') or 
                    response_data.get('output') or 
                    'Let me help you with that.'
                )
                
                conv_data['messages'].append({
                    'role': 'assistant',
                    'content': ai_response,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'n8n_agent'
                })
                
                return jsonify({
                    'response': ai_response,
                    'conversation_id': conversation_id,
                    'transferred_to_agent': True,
                    'action_type': response_data.get('action_type', 'scheduling')
                })
            else:
                # Fallback
                ai_response = "I'm having trouble with the scheduling system. Could you provide your email and preferred time, and I'll have someone reach out to you directly?"
                
                conv_data['messages'].append({
                    'role': 'assistant',
                    'content': ai_response,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'fallback'
                })
                
                return jsonify({
                    'response': ai_response,
                    'conversation_id': conversation_id,
                    'transferred_to_agent': False
                })
        
        # Regular conversation with local LLM + scheduling offer
        else:
            # Get response from local LLM
            local_response = get_local_ai_response_with_history(conv_data)
            
            # Always add scheduling offer at the end
            enhanced_response = f"{local_response}\n\nWould you like to schedule a brief consultation to learn more about how this could work for your specific situation?"
            
            # Add to conversation history
            conv_data['messages'].append({
                'role': 'assistant',
                'content': enhanced_response,
                'timestamp': datetime.now().isoformat(),
                'source': 'local_llm'
            })
            
            return jsonify({
                'response': enhanced_response,
                'conversation_id': conversation_id,
                'awaiting_scheduling_response': True,
                'ask_for_scheduling': True
            })
    
    except Exception as e:
        logger.error(f"Error in chat route: {str(e)}")
        return jsonify({'error': 'Something went wrong. Please try again.'}), 500

def create_conversation_summary(conv_data):
    """Create a brief summary of the conversation for n8n"""
    messages = conv_data['messages']
    user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
    
    if len(user_messages) > 1:
        return f"User asked about: {user_messages[0][:100]}... Latest: {user_messages[-1][:100]}"
    elif len(user_messages) == 1:
        return f"User asked: {user_messages[0][:100]}"
    else:
        return "User interested in learning more"

def is_affirmative_response(message):
    """Check if user response is affirmative for scheduling"""
    message_lower = message.lower().strip()
    
    affirmative_keywords = [
        'yes', 'yeah', 'yep', 'sure', 'okay', 'ok', 'definitely', 'absolutely',
        'please', 'schedule', 'book', 'i would like', 'sounds good', 'that works',
        'let\'s do it', 'go ahead', 'i\'m interested', 'sign me up', 'why not'
    ]
    
    negative_keywords = [
        'no', 'nope', 'not now', 'maybe later', 'not interested', 'no thanks',
        'not today', 'i\'ll think about it', 'not right now'
    ]
    
    # Check for negative responses first
    if any(keyword in message_lower for keyword in negative_keywords):
        return False
    
    # Check for affirmative responses
    if any(keyword in message_lower for keyword in affirmative_keywords):
        return True
    
    # Default to False if unclear
    return False

def get_local_ai_response_with_history(conv_data):
    """Get AI response from local Ollama with conversation history"""
    try:
        messages = conv_data['messages']
        
        # Build conversation context (last 6 messages for efficiency)
        conversation_text = ""
        for msg in messages[-6:]:
            if msg['role'] == 'user':
                conversation_text += f"Human: {msg['content']}\n"
            elif msg['role'] == 'assistant' and msg.get('source') == 'local_llm':
                # Only include local LLM responses, not scheduling responses
                clean_content = msg['content'].split('\n\nWould you like to schedule')[0]  # Remove scheduling offer
                conversation_text += f"Assistant: {clean_content}\n"
        
        system_prompt = """You are an AI assistant for Optivus, an AI document intelligence and automation platform.

Key information about Optivus:
- We implement AI systems that process documents, extract data, and automate workflows
- Our solutions provide 300-500% ROI within the first year  
- We help businesses save 80% of time spent on document processing
- We work with companies in finance, healthcare, legal, manufacturing, and other industries
- Our platform handles PDFs, Word docs, Excel files, emails, and scanned documents
- We integrate with existing business systems and provide 24/7 automation

Your role:
- Answer questions about AI, automation, and business process improvement
- Explain how document intelligence can solve specific business problems
- Be helpful and knowledgeable about AI/ML technologies
- Build naturally on the conversation history
- Keep responses concise but informative (2-4 sentences typically)
- Focus on business value and practical applications
- DO NOT mention scheduling or meetings - the system handles that separately

Recent conversation context:
""" + conversation_text

        current_message = messages[-1]['content']
        full_prompt = f"{system_prompt}\n\nHuman: {current_message}\n\nAssistant:"
        
        response = requests.post(
            OLLAMA_URL,
            json={
                'model': 'llama3.2',
                'prompt': full_prompt,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 200
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            ai_response = response.json().get('response', '').strip()
            if ai_response:
                return ai_response
            else:
                return "I'd be happy to help you understand how AI document intelligence can transform your business operations!"
        else:
            logger.error(f"Ollama error: {response.status_code}")
            return "I'm here to help you understand how AI can streamline your business processes."
    
    except Exception as e:
        logger.error(f"Error getting local AI response: {str(e)}")
        return "I'm here to help you learn about AI automation solutions for your business."

def send_to_n8n_ai_agent(message, conversation_id, conv_data):
    """Send message to N8N AI agent"""
    try:
        payload = {
            'message': message,
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'source': 'website_chat_transfer'
        }
        
        logger.info(f"Sending to N8N: {message[:50]}...")
        
        response = requests.post(
            N8N_AI_AGENT_WEBHOOK,
            json=payload,
            timeout=30
        )
        
        logger.info(f"N8N response status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                logger.info(f"Raw N8N JSON Response: {result}")
                
                # Handle if n8n returns an array (list) instead of object
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]  # Take the first item
                
                return {'success': True, 'data': result}
            except Exception as e:
                logger.error(f"Failed to parse N8N JSON: {e}")
                logger.info(f"N8N Raw Text Response: {response.text}")
                return {
                    'success': True, 
                    'data': {
                        'message': response.text,
                        'action_type': 'scheduling'
                    }
                }
        else:
            logger.error(f"N8N error: {response.status_code}")
            return {'success': False}
            
    except Exception as e:
        logger.error(f"Error sending to N8N: {str(e)}")
        return {'success': False}

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_conversations': len(conversation_history),
        'services': {
            'n8n_ai_agent': check_n8n_availability(),
            'ollama': check_ollama_availability()
        }
    })

def check_n8n_availability():
    try:
        # Just check if the URL is reachable without sending data
        response = requests.get(N8N_AI_AGENT_WEBHOOK.replace('/webhook-test/', '/webhook-health/'), timeout=5)
        return True  # If no exception, assume it's available
    except:
        return False

def check_ollama_availability():
    try:
        response = requests.get(OLLAMA_URL.replace('/api/generate', '/api/tags'), timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == '__main__':
    logger.info("🚀 Starting Streamlined Flask app...")
    logger.info(f"🤖 N8N AI Agent: {N8N_AI_AGENT_WEBHOOK}")
    logger.info(f"🧠 Local LLM: {OLLAMA_URL}")
    logger.info("📅 Every response now includes scheduling offer!")
    
    app.run(debug=True, host='0.0.0.0', port=8000)