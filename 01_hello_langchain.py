"""LangChain 入门示例 — 第一个 Chain"""
import sys
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. 初始化模型
# llm = ChatOpenAI(
#     model="deepseek-v4-pro",
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url=os.getenv("OPENAI_BASE_URL"),
# )

llm2= init_chat_model(
    model="deepseek-v4-pro",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),

)

# 2. 创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手，用中文回答问题。"),
    ("user", "{topic}"),
])

# # 3. 用 LCEL (|) 串联成 Chain
# chain = prompt | llm

if __name__ == "__main__":
    reps= llm2.invoke("你是谁")
    print(reps)

# # 4. 调用
# result = chain.invoke({"topic": "什么是 LangChain？请用一两句话简单介绍。"})
# print(result.content)
