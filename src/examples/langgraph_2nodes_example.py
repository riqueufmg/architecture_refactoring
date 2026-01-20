import os
from dotenv import load_dotenv
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

class State(TypedDict):
    question: str
    answer: str

llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0.7,
    api_key=os.environ["OPENAI_API_KEY"],
)

def answer_node(state: State):
    res = llm.invoke([HumanMessage(content=state["question"])])
    return {"answer": res.content}

def refine_node(state: State):
    prompt = f"Improve this answer:\n{state['answer']}"
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": res.content}

graph = StateGraph(State)
graph.add_node("answer", answer_node)
graph.add_node("refine", refine_node)

graph.set_entry_point("answer")
graph.add_edge("answer", "refine")
graph.add_edge("refine", END)

app = graph.compile()

out = app.invoke({"question": "Explain LangGraph simply", "answer": ""})
print(out["answer"])
