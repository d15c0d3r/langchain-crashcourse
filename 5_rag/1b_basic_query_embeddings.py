from utils.vector_db import get_vector_db

# For RAG - Retrieval-Augmented Generation
# 1. Create embeddings
# 2. Store embeddings in a vector store
# 3. Retrieve relevant embeddings from the vector store
# 4. Use the retrieved embeddings to generate a response

# 3, 4 are covered in this file
db = get_vector_db()

query = "Who is Odysseus' wife?"

# retrieve relevant documents from the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 7}
)

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\nDocument {i+1}: \n{doc.page_content}\n")