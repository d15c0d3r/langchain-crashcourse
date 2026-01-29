from langchain_core.messages import AIMessage, HumanMessage
from common.model import get_model

model = get_model()

chatHistory = []

systemMessage = "You are a helpful assistant that can answer questions and help with tasks."
chatHistory.append(systemMessage)

while True:
    userInput = input("You: ")
    chatHistory.append(HumanMessage(content=userInput))
    if userInput.lower() == "exit":
        break

    result = model.invoke(chatHistory)
    response = result.content
    chatHistory.append(AIMessage(content=response))

    print(f"Assistant: {response}")

print("----- Message History -----")
print(chatHistory)