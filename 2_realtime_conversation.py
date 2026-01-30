from langchain_core.messages import AIMessage, HumanMessage
from common.model import get_model

model = get_model()

chat_history = []

system_message = "You are a helpful assistant that can answer questions and help with tasks."
chat_history.append(system_message)

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break

    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f"Assistant: {response}")

print("----- Message History -----")
print(chat_history)
