from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common.model import get_model

model = get_model()

messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {num_jokes} jokes"),
]

prompt_template = ChatPromptTemplate.from_messages(messages)

# Custom lambda function to uppercase the output
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word Count: {len(x.split())}\n{x}")

# Think of RunnableLambda as "turn any Python function into a Runnable so it can participate in a chain with |." What happens inside that function is entirely up to you.
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

result = chain.invoke({"topic": "doctors", "num_jokes": 3})
print(result)
