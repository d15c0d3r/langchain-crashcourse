from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

_model = ChatOpenAI(
    model="liquid/lfm-2.5-1.2b-thinking:free",  # OpenRouter model slug
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def get_model():
    return _model
