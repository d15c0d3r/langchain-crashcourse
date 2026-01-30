from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from common.model import get_model

## Type 1: from_template() - Normal prompt template. By default, the first message is the system message.
model = get_model()

template = "Tell me an average weight of {thing} in {units}."
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"thing": "elephant", "units": "kilograms"})
print("Type 1: ",prompt)

# Type 2: from_messages() - Define messages as a list of tuples with (role, content).
# first role should be "system" or "human"
messages = [("system", "You are a comedian who tells jokes about {topic}"),
 ("human", "Tell me {num_jokes} jokes")]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "elephants", "num_jokes": 3})
print("Type 2: ", prompt)

# Doesn't work
messages = [("system", "You are a comedian who tells jokes about {topic}"),
 HumanMessage(content="Tell me {num_jokes} jokes")]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "elephants", "num_jokes": 3})
print("Type 3: ", prompt)
# Output:
# messages=[SystemMessage(content='You are a comedian who tells jokes about elephants', additional_kwargs={}, response_metadata={}), HumanMessage(content='Tell me {num_jokes} jokes', additional_kwargs={}, response_metadata={})]