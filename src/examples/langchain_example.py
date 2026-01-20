import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="xiao­mi/mimo-v2-flash",
    temperature=0.7,
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

prompt = ChatPromptTemplate.from_template(
    "Summarize the following text with less than 5 words:\n{text}"
)

chain = prompt | llm

result = chain.invoke({
    "text": "LangChain helps developers build AI apps faster."
}).content

print(result)