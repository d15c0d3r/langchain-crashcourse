from typing import TypedDict, List

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from common.model import get_model
from utils.vector_db import get_chroma_db

# --- Setup ---

db = get_chroma_db("chroma_db_with_metadata")
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = get_model()

# --- Prompts ---

# This prompt helps the AI reformulate the question based on chat history
# so it becomes a standalone question that the retriever can understand.

# NOTE:
# 1. The input of MessagesPlaceholder should the be field_name of the state,
#    in this case, the State is RAGState and the fieldname is chat_history.
# 2. MessagesPlaceholder must come BEFORE ("human", "{input}") so the LLM sees history before the latest question.
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question "
     "which might reference context in the chat history, "
     "formulate a standalone question which can be understood "
     "without the chat history. Do NOT answer the question, just "
     "reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# This prompt helps the AI provide concise answers based on retrieved context.
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant for question-answering tasks. Use "
     "the following pieces of retrieved context to answer the "
     "question. If you don't know the answer, just say that you "
     "don't know. Use three sentences maximum and keep the answer "
     "concise.\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# --- State: shared data that flows through the graph ---

class RAGState(TypedDict):
    input: str
    chat_history: List
    context: str
    answer: str


# --- Node functions: each node does one step of the RAG pipeline ---

def contextualize(state: RAGState) -> dict:
    """Reformulate the question using chat history so it's standalone."""
    if state["chat_history"]:
        chain = contextualize_q_prompt | llm | StrOutputParser()
        reformulated = chain.invoke({
            "chat_history": state["chat_history"],
            "input": state["input"],
        })
        return {"input": reformulated}
    return {}


def retrieve(state: RAGState) -> dict:
    """Retrieve relevant documents from the vector store."""
    docs = retriever.invoke(state["input"])
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"context": context}


def generate(state: RAGState) -> dict:
    """Generate an answer using the retrieved context."""
    chain = qa_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": state["context"],
        "chat_history": state["chat_history"],
        "input": state["input"],
    })
    return {"answer": answer}


# --- Build the graph: contextualize -> retrieve -> generate ---

graph = StateGraph(RAGState)
graph.add_node("contextualize", contextualize)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge(START, "contextualize")
graph.add_edge("contextualize", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

rag_chain = graph.compile()


# --- Chat loop ---

def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    # we dont iniialize a RAGState object, instead we use chat_history for persistence.
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        result = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history,
            "context": "",
            "answer": "",
        })
        print(f"AI: {result['answer']}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["answer"]))


if __name__ == "__main__":
    continual_chat()
