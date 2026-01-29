from common.model import get_model

model = get_model()

result = model.invoke("What is 81 divided by 9?")
print(result.content)
