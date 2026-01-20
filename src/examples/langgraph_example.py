import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    return a * b

llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0.1,
    api_key=os.environ["OPENAI_API_KEY"],
)

agent_executor = create_agent(
    model=llm,
    tools=[multiply],
    system_prompt="You are a helpful assistant. Use tools when needed.",
)

graph = StateGraph(dict)
graph.add_node("agent", agent_executor)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

app = graph.compile()

response = app.invoke({
    "messages": [
        {"role": "user", "content": "Use the multiply tool to get 8 times 7"}
    ]
})

print(response["messages"][-1].content)