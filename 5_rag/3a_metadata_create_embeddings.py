import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from utils.vector_db import get_chroma_db

# 3a and 3b are basically same as 2a and 2b but we provide metadata to the documents.
# This is useful when we want to retrieve documents based on their metadata.
# This also helps in improving the accuracy of the results and the embedding model doesnt hallucinate.
db_dir = "chroma_db_with_metadata"
current_dir = os.path.dirname(os.path.abspath(__file__))
files_dir = os.path.join(current_dir, "books")

if not os.path.exists(files_dir):
    print("\nFiles directory does not exist")
    exit(1)

print("\nCreating chroma database with metadata...")
book_files = [f for f in os.listdir(files_dir) if f.endswith(".txt")]

all_documents = []
for book_file in book_files:
    book_path = os.path.join(files_dir, book_file)
    loader = TextLoader(book_path)
    book_docs = loader.load()
    for doc in book_docs:
        doc.metadata = {"source": book_file}
        all_documents.append(doc)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=200)
doc_chunks = text_splitter.split_documents(all_documents)

db = get_chroma_db(db_dir)
batch_size = 5000
for i in range(0, len(doc_chunks), batch_size):
    db.add_documents(doc_chunks[i:i + batch_size])

print(f"\nAdded {len(doc_chunks)} documents to the database")
