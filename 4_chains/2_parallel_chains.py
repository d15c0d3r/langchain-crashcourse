from langchain_core.runnables import RunnableLambda, RunnableParallel
from common.model import get_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = get_model()
# As the name suggests, The input of the previous chain is going to the output of all the parallel
# chains.
# create prompt template to review a product
# -> system: You are an expert product reviewer
# -> human: Review the product {product}
# create pros and cons runnable parallel lambda functions
# the output of the model will be input for both pros and cons
# create a chain -> PromptTemplate -> Model -> StrOutputParser() -> RunnableParallel(pros, cons) -> CombineProsAndCons()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert product reviewer"),
    ("human", "What are the features of {product}"),
])


def analyze_pros(features):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert product reviewer"),
        ("human",
         "Given these {features}, what are the pros of these features"),
    ])
    return prompt_template.format_prompt(features=features)


def analyze_cons(features):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert product reviewer"),
        ("human",
         "Given these {features}, what are the cons of these features"),
    ])
    return prompt_template.format_prompt(features=features)


pros_chain = RunnableLambda(analyze_pros) | model | StrOutputParser()
cons_chain = RunnableLambda(analyze_cons) | model | StrOutputParser()


def combine_pros_and_cons(pros, cons):
    return f"Pros: {pros}\n\nCons: {cons}"


chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(pros=pros_chain, cons=cons_chain)
    # RunnableParallel outputs a dictionary where the keys match the keyword argument names you passed when constructing it.
    | RunnableLambda(lambda x: combine_pros_and_cons(x["pros"], x["cons"]))
)

result = chain.invoke({"product": "iPhone 16"})
print(result)
