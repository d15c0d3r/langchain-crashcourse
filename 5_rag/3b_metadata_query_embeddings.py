import os

from utils.vector_db import get_chroma_db

db_dir = "chroma_db_with_metadata"
current_dir = os.path.dirname(os.path.abspath(__file__))
files_dir = os.path.join(current_dir, "books")

if not os.path.exists(files_dir):
    print("\nFiles directory does not exist")
    exit(1)

db = get_chroma_db(db_dir)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

docs = retriever.invoke("How did Juliet die?")

for i, doc in enumerate(docs):
    print(f"\nDocument {i+1}: \n{doc.page_content}\n")
