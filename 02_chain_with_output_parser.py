"""LangChain 示例 — 带输出解析器的 Chain，导入 01 的模块"""
import sys
import os
import importlib
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

# 导入以数字开头的模块（如 01_hello_langchain）
# Python 的 import 语句不支持数字开头的标识符，需要用 importlib
m01 = importlib.import_module("01_hello_langchain")
llm2 = m01.llm2  # 直接使用 01 中已初始化的模型

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手，用中文回答。"),
    ("user", "{question}"),
])

# LCEL: prompt -> llm2 -> output parser
chain = prompt | llm2 | StrOutputParser()

result = chain.invoke({"question": "用一句话解释什么是 LCEL？"})
print(result)
