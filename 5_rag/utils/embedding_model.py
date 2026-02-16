from langchain_ollama import OllamaEmbeddings

embedding_model = OllamaEmbeddings(model="bge-m3")

def get_embedding_model():
    return embedding_model