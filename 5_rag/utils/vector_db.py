from langchain_chroma import Chroma
from utils.embedding_model import get_embedding_model

vector_db = None
vector_db_directory = "/Users/bcdev66/Desktop/langchain/5_rag/vector_db"


def get_vector_db():
    global vector_db, vector_db_directory
    if vector_db:
        return vector_db
    embedding_model = get_embedding_model()
    # creates a new DB if the directory is missing, or loads the existing one if it's there.
    # It just needs to always be called.
    vector_db = Chroma(persist_directory=vector_db_directory,
                       embedding_function=embedding_model)
    return vector_db
