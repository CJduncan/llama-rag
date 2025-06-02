# upload_guide.py
from app import rag_system

# Upload the guide document
success = rag_system.add_document('ai_document_guide.txt', 'AI Document Intelligence Business Guide')

if success:
    print("✅ Successfully uploaded the business guide!")
    print(f"Document count: {rag_system.get_document_count()}")
else:
    print("❌ Failed to upload the document")