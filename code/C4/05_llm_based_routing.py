import os
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableBranch
from dotenv import load_dotenv
load_dotenv()
# llm = ChatDeepSeek(
#     model="deepseek-chat", 
#     temperature=0, 
#     api_key=os.getenv("DEEPSEEK_API_KEY")
#     )
# llm = ChatOpenAI(
#     # model="glm-4.7-flash-free",
#     model="gpt-4.1-free",
#     temperature=0,
#     max_tokens=4096,
#     api_key=os.getenv("AIHUBMIX_API_KEY"),
#     base_url="https://aihubmix.com/v1"
# )
llm = ChatOpenAI(
    # model="glm-4.7-flash-free",
    # model="MiniMax-M2.5",
    model="deepseek-ai/DeepSeek-V3",
    temperature=0,
    max_tokens=4096,
    api_key=os.getenv("SILICON_FLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)
# 1. 设置不同菜系的处理链
sichuan_prompt = ChatPromptTemplate.from_template(
    "你是一位川菜大厨。请用正宗的川菜做法，回答关于「{question}」的问题。"
)
sichuan_chain = sichuan_prompt | llm | StrOutputParser()

cantonese_prompt = ChatPromptTemplate.from_template(
    "你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{question}」的问题。"
)
cantonese_chain = cantonese_prompt | llm | StrOutputParser()

# 定义备用通用链
general_prompt = ChatPromptTemplate.from_template(
    "你是一个美食助手。请回答关于「{question}」的问题。"
)
general_chain = general_prompt | llm | StrOutputParser()


# 2. 创建路由链
classifier_prompt = ChatPromptTemplate.from_template(
    """根据用户问题中提到的菜品，将其分类为：['川菜', '粤菜', 或 '其他']。
    不要解释你的理由，只返回一个单词的分类结果。
    问题: {question}"""
)
classifier_chain = classifier_prompt | llm | StrOutputParser()

# 定义路由分支
router_branch = RunnableBranch(
    (lambda x: "川菜" in x["topic"], sichuan_chain),
    (lambda x: "粤菜" in x["topic"], cantonese_chain),
    general_chain  # 默认选项
)

# 组合成完整路由链
full_router_chain = {"topic": classifier_chain, "question": lambda x: x["question"]} | router_branch
print("完整的路由链创建成功。\n")


# 3. 运行演示查询
demo_questions = [
    {"question": "麻婆豆腐怎么做？"},      # 应该路由到川菜
    {"question": "白切鸡的正宗做法是什么？"}, # 应该路由到粤菜
    {"question": "番茄炒蛋需要放糖吗？"}      # 应该路由到其他
]

total_start_time = time.time()

for i, item in enumerate(demo_questions, 1):
    question = item["question"]
    print(f"\n--- 问题 {i}: {question} ---")
    question_start = time.time()

    try:
        # 获取路由决策
        print("  - 分类器调用中...")
        classifier_start = time.time()
        topic = classifier_chain.invoke({"question": question})
        classifier_time = time.time() - classifier_start
        print(f"路由决策: {topic} (耗时: {classifier_time:.2f}秒)")

        # 执行完整链 - 使用流式输出
        print("  - 生成回答中(流式输出)...")
        print("  回答: ", end="", flush=True)
        answer_start = time.time()
        
        # result = full_router_chain.invoke(item)
        # 方式1: 使用 .stream() 获取流式输出
        stream = full_router_chain.stream(item)
        result = ""
        for chunk in stream:
            print(chunk, end="", flush=True)
            result += chunk

        answer_time = time.time() - answer_start
        print()  # 换行
        print(f"  回答生成耗时: {answer_time:.2f}秒")
        # print(f"完整回答: {result}")  # 如需查看完整回答可取消注释
    except Exception as e:
        print(f"执行错误: {e}")

    question_time = time.time() - question_start
    print(f"问题总耗时: {question_time:.2f}秒")

total_time = time.time() - total_start_time
print(f"\n{'='*60}")
print(f"总执行时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
print(f"平均每个问题: {total_time/len(demo_questions):.2f}秒")
print(f"{'='*60}")

