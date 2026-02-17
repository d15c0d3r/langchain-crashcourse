from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from common.model import get_model
from utils.vector_db import get_chroma_db

# get the vector store instance

db = get_chroma_db("chroma_db_with_metadata")

# create a retriever with a search_type and search_kwargs

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
query = "How can I learn more about Langchain?"

# invoke the retriever with the query and get the relevant documents
docs = retriever.invoke(query)

# create a input prompt to the model
combined_input = (
    "Here are the documents that might help answer the question: "
    + "Documents: \n\n".join([doc.page_content for doc in docs])
    + f"\n\n Question: {query}"
    + "\n Please provide the answer only based on the documents provided. if the answer is not found, "
    + "please say that you don't know the answer."
)

messages = [
    ("system", "You are a helpful assistant"),
    ("human", combined_input),
]
prompt_template = ChatPromptTemplate.from_messages(messages)

# invoke the model with the input and get the response
model = get_model()
chain = prompt_template | model | StrOutputParser()
response = chain.invoke({})
print(response)