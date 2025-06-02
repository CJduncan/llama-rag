from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import chromadb
import ollama
import os
import hashlib
from typing import List, Dict
import PyPDF2
import docx
from pathlib import Path
import time
from werkzeug.utils import secure_filename
import secrets
import logging

app = Flask(__name__)
# Secure configuration for production
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', secrets.token_hex(32)),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    UPLOAD_FOLDER='uploads',
    SESSION_COOKIE_SECURE=os.environ.get('FLASK_ENV') == 'production',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax'
)

# Configure CORS for production
CORS(app, origins=["http://localhost:5000", "https://yourdomain.com"])

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)

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
    """Main RAG system using Ollama and ChromaDB"""
    
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path="./chromadb")
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.doc_processor = DocumentProcessor()
        
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
            
            # Add to ChromaDB
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
    
    def generate_response(self, query: str, context: str, model: str = "llama3.2") -> str:
        """Generate response using Ollama with context"""
        try:
            prompt = f"""You are a helpful AI assistant for a document intelligence platform. Answer the user's question directly and concisely using the provided context. Keep responses under 3 sentences when possible. Do not mention "based on the document" or "according to the context" - just provide the information naturally.

Context: {context}

Question: {query}

Provide a direct, conversational answer:"""
            
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            
            return response['message']['content']
            
        except Exception as e:
            app.logger.error(f"Error generating response: {e}")
            return f"I'm sorry, I'm having trouble processing your request right now. Please try again later."
    
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

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        query = data.get('message', '').strip()
        model = data.get('model', 'llama3.2')
        
        if not query:
            return jsonify({'error': 'No message provided'}), 400
        
        # Rate limiting check could go here in production
        
        # Search for relevant documents
        search_results = rag_system.search_documents(query)
        
        if search_results:
            # Combine context from search results
            context = "\n\n".join([result['content'] for result in search_results])
            
            # Generate response
            response = rag_system.generate_response(query, context, model)
            
            return jsonify({
                'response': response,
                'sources': len(search_results)  # Don't expose full source content to public
            })
        else:
            return jsonify({
                'response': "I don't have any relevant documents to answer your question at the moment. This AI system works best when it has access to your business documents for context.",
                'sources': 0
            })
            
    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Sorry, there was an error processing your request.'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        doc_count = rag_system.get_document_count()
        return jsonify({
            'status': 'healthy',
            'document_count': doc_count,
            'timestamp': int(time.time())
        })
    except Exception as e:
        app.logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
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
    # Production vs Development settings
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    port = int(os.environ.get('PORT', 5000))
    
    if debug_mode:
        print("🚀 Starting Flask RAG System...")
        print(f"📱 Main page: http://localhost:{port}/")
        print("⚠️  Running in development mode")
    
    app.run(
        debug=debug_mode,
        host='0.0.0.0',
        port=port
    )