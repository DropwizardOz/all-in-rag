import os
import time
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnablePassthrough
from langchain_community.utils.math import cosine_similarity
import numpy as np

# 1. 定义路由描述
sichuan_route_prompt = "你是一位处理川菜的专家。用户的问题是关于麻辣、辛香、重口味的菜肴，例如水煮鱼、麻婆豆腐、鱼香肉丝、宫保鸡丁、花椒、海椒等。"
cantonese_route_prompt = "你是一位处理粤菜的专家。用户的问题是关于清淡、鲜美、原汁原味的菜肴，例如白切鸡、老火靓汤、虾饺、云吞面等。"

route_prompts = [sichuan_route_prompt, cantonese_route_prompt]
route_names = ["川菜", "粤菜"]

# 初始化嵌入模型，并对路由描述进行向量化
print("正在加载嵌入模型...")
start_time = time.time()
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    encode_kwargs={'normalize_embeddings': True}  # 归一化以提高相似度计算的稳定性
)
print("嵌入模型加载完成，正在计算路由提示的嵌入...")
route_prompt_embeddings = embeddings.embed_documents(route_prompts)
init_time = time.time() - start_time
print(f"已定义 {len(route_names)} 个路由: {', '.join(route_names)}")
print(f"初始化耗时: {init_time:.2f}秒\n")

# 2. 定义不同路由的目标链
# llm = ChatDeepSeek(
#     model="deepseek-chat", 
#     temperature=0, 
#     api_key=os.getenv("DEEPSEEK_API_KEY")
#     )
# llm = ChatOpenAI(
#     # model="glm-4.7-flash-free",
#     # model="MiniMax-M2.5",
#     model="DeepSeek-R1-0528",
#     # model="Qwen3-30B-A3B", # 30B 模型推理速度远慢于 7B/13B 模型
#     temperature=0,
#     max_tokens=4096,
#     api_key=os.getenv("CSNET_API_KEY"),
#     base_url="https://api.scnet.cn/api/llm/v1"
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
# 定义川菜和粤菜处理链
sichuan_chain = (
    PromptTemplate.from_template("你是一位川菜大厨。请用正宗的川菜做法，回答关于「{query}」的问题。")
    | llm
    | StrOutputParser()
)
cantonese_chain = (
    PromptTemplate.from_template("你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{query}」的问题。")
    | llm
    | StrOutputParser()
)

route_map = { "川菜": sichuan_chain, "粤菜": cantonese_chain }
print("川菜和粤菜的处理链创建成功。\n")

# 3. 创建路由函数
def route(info):
    # 对用户查询进行嵌入
    print("  - 计算查询嵌入...")
    embed_start = time.time()
    query_embedding = embeddings.embed_query(info["query"])
    embed_time = time.time() - embed_start
    print(f"  - 嵌入计算耗时: {embed_time:.3f}秒")

    # 计算与各路由提示的余弦相似度
    print("  - 计算相似度...")
    similarity_start = time.time()
    similarity_scores = cosine_similarity([query_embedding], route_prompt_embeddings)[0]
    similarity_time = time.time() - similarity_start
    print(f"  - 相似度计算耗时: {similarity_time:.3f}秒")

    # 找到最相似的路由
    chosen_route_index = np.argmax(similarity_scores)
    chosen_route_name = route_names[chosen_route_index]

    print(f"路由决策: 检测到问题与'{chosen_route_name}' 最相似 (相似度: {similarity_scores[chosen_route_index]:.3f})")

    # 获取对应的处理链
    chosen_chain = route_map[chosen_route_name]

    # 直接调用选中的链并返回结果
    return chosen_chain.invoke(info)

# 创建完整的路由链
full_chain = RunnableLambda(route)


# 4. 运行演示查询
demo_queries = [
    "水煮鱼怎么做才嫩？",        # 应该路由到川菜
    "如何做一碗清淡的云吞面？",    # 应该路由到粤菜
    "麻婆豆腐的核心调料是什么？",  # 应该路由到川菜
]

total_start_time = time.time()

for i, query in enumerate(demo_queries, 1):
    print(f"\n--- 问题 {i}: {query} ---")
    query_start = time.time()

    # 添加请求间隔,避免触发速率限制
    if i > 1:
        print("等待2秒,避免触发API速率限制...")
        time.sleep(2)

    try:
        # 传入字典，full_chain 会直接返回最终答案
        result = full_chain.invoke({"query": query})
        query_time = time.time() - query_start
        print(f"回答: {result}")
        print(f"总耗时: {query_time:.2f}秒")
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "Rate Limit" in error_msg:
            print(f"❌ API速率限制错误!")
            print(f"💡 建议:")
            print(f"   - 等待1-2分钟后重试")
            print(f"   - 或切换到其他API提供商")
            print(f"   - 或减少并发请求数量")
            print(f"错误详情: {e}")
            break  # 遇到速率限制时停止执行
        else:
            print(f"执行错误: {e}")
        query_time = time.time() - query_start

total_time = time.time() - total_start_time
print(f"\n{'='*60}")
print(f"总执行时间: {total_time:.2f}秒")
print(f"平均每个问题: {total_time/len(demo_queries):.2f}秒")
print(f"{'='*60}")

