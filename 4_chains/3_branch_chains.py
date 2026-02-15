from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from common.model import get_model

# As the name suggests, Branching chains is basically a chain that dynamically branches our to
# different chains based on the type of the input.
# Unlike parallel chains, branching chains are like switch statements. They are basically
# if else statements mapping to the right branch based on the input. But a branch is a chain itself.

model = get_model()

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant"),
        ("human",
         "Generate a Thank you note for the positive feedback: {feedback}")
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant"),
        ("human",
         "Generate a note addressing the negative feedback: {feedback}")
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant"),
        ("human",
         "Generate a request to provide more details on the neutral feedback: {feedback}")
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant"),
        ("human",
         "Generate a message to escalate the feedback to a human agent: {feedback}")
    ]
)

feedback_classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant"),
        ("human",
         "Classify the feedback into Positive, Negative, Neutral or Escalate: {feedback}")
    ]
)

feedback_branches = RunnableBranch(
    (
        lambda x: "Positive" in x["classification"],
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "Negative" in x["classification"],
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "Neutral" in x["classification"],
        neutral_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "Escalate" in x["classification"],
        escalate_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()
)

feedback_classification_chain = feedback_classification_template | model | StrOutputParser()

# RunnablePassthrough is used to pass the input to the next chain.
# the "feedback" key will be only passed to the main chain and not to the branches,
# but with the RunnableLambda, we can pass the input to the branches.
chain = RunnablePassthrough.assign(
    classification=feedback_classification_chain
) | feedback_branches

# Example feedbacks
# Good feedback - "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad feedback - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral feedback - "The product is okay. It works as expected but nothing exceptional."
# Escalate feedback - "I'm not sure about the product yet. I want to talk to a human agent."
customer_feedback = "The product is okay. It works as expected but nothing exceptional."
result = chain.invoke({"feedback": customer_feedback})
print(result)
