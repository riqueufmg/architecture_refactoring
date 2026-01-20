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

def is_good(state):
    if len(state["answer"]) < 50:
        return "retry"
    return "done"

def answer_node(state: State):
    res = llm.invoke([HumanMessage(content=state["question"])])
    return {"answer": res.content}

def refine_node(state: State):
    prompt = f"Improve this answer with more than 60 characters:\n{state['answer']}"
    return {"question": prompt}

graph = StateGraph(State)

graph.add_node("answer", answer_node)
graph.add_node("refine", refine_node)

graph.set_entry_point("answer")

graph.add_conditional_edges(
    "answer",
    is_good,
    {
        "retry": "refine",
        "done": END
    }
)

graph.add_edge("refine", "answer")

app = graph.compile()

out = app.invoke({"question": "Explain LangGraph with at maximum 5 words", "answer": ""})
print(out["answer"])
