from langchain_ollama import OllamaEmbeddings


def get_embedding_model():
    embedding_model = OllamaEmbeddings(model="qwen3-embedding")
    return embedding_model
