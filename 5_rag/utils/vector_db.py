import os

from langchain_chroma import Chroma
from utils.embedding_model import get_embedding_model

parent_directory = "/Users/bcdev66/Desktop/langchain/5_rag/vector_db"


def get_chroma_db(dir):
    chroma_db_directory = os.path.join(parent_directory, dir)
    # creates a new DB if the directory is missing, or loads the existing one if it's there.
    # It just needs to always be called.
    chroma_db = Chroma(
        persist_directory=chroma_db_directory,
        embedding_function=get_embedding_model(),
    )
    return chroma_db
