import os
from dotenv import load_dotenv
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

load_dotenv()

class State(TypedDict):
    question: str
    answer: str

llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0.7,
    api_key=os.environ["OPENAI_API_KEY"],
)

def route(state):
    if "code" in state["question"].lower():
        return "code"
    return "explain"

def router_node(state: State):
    return {}

def explain_node(state):
    res = llm.invoke([HumanMessage(content=state["question"])])
    return {"answer": res.content}

def code_node(state):
    res = llm.invoke([HumanMessage(content=f"Write code for: {state['question']}")])
    return {"answer": res.content}

graph = StateGraph(State)

graph.add_node("router", router_node)
graph.add_node("explain", explain_node)
graph.add_node("code", code_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    route,
    {
        "code": "code",
        "explain": "explain"
    }
)

graph.add_edge("explain", END)
graph.add_edge("code", END)

app = graph.compile()

out = app.invoke({"question": "I need to understand Fibonacci", "answer": ""})
print(out["answer"])