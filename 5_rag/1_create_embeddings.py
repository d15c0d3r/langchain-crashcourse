import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.vector_db import get_vector_db

# For RAG - Retrieval-Augmented Generation
# 1. Create embeddings
# 2. Store embeddings in a vector store
# 3. Retrieve relevant embeddings from the vector store
# 4. Use the retrieved embeddings to generate a response

# 1, 2 are covered in this file

current_directory = os.path.dirname(os.path.abspath(__file__))
file_directory = os.path.join(current_directory, "books", "odyssey.txt")

if not os.path.exists(file_directory):
    print("\nFile does not exist")
    exit(1)

loader = TextLoader(file_directory)
documents = loader.load()

# split the focused documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=400)
chunks = text_splitter.split_documents(documents)
print(f"\nNumber of chunks: {len(chunks)}")

# create embeddings using vector_db (Chroma)
vector_db = get_vector_db()
vector_db.add_documents(
    chunks
)

print(f"Vector database: {vector_db}")
