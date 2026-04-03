import os
import asyncio
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.llms.deepseek import DeepSeek
# from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    BatchEvalRunner,
)
from llama_index.core.evaluation.eval_utils import get_results_df
from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset

# Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.llm = DeepSeek(model="deepseek-ai/DeepSeek-V3", temperature=0.1, 
    api_key=os.getenv("SILICON_FLOW_API_KEY"),
    api_base="https://api.siliconflow.cn/v1")

# Settings.llm = OpenAI(
#     model="deepseek-ai/DeepSeek-V3",
#     temperature=0.1,
#     api_key=os.getenv("SILICON_FLOW_API_KEY"),
#     api_base="https://api.siliconflow.cn/v1"
# )
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en", device="cpu")
print("✓ 模型初始化完成")

async def main():
    print(f"\n{'='*20} 开始执行 {'='*20}")
    total_start = time.time()

    # 1. 加载文档
    print("加载文档中...")
    load_start = time.time()
    reader = SimpleDirectoryReader(input_files=["../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"])
    documents = reader.load_data()
    load_time = time.time() - load_start
    print(f"✓ 加载了 {len(documents)} 个文档 ({load_time:.1f}秒)")

    # 1.1 加载或生成响应评估数据集
    if os.path.exists("./c6_response_eval_dataset.json"):
        print("加载响应评估数据集...")
        response_eval_dataset = QueryResponseDataset.from_json("./c6_response_eval_dataset.json")
    else:
        print("生成响应评估数据集...")
        dataset_generator = DatasetGenerator.from_documents(documents[:10])  # 减少文档数量
        response_eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=15)  # 减少问题数量
        response_eval_dataset.save_json("./c6_response_eval_dataset.json")



    # 2. 构建两种不同的RAG查询引擎和检索器进行对比
    # 2.1 句子窗口检索
    print("\n构建句子窗口检索...")
    sentence_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_nodes = sentence_parser.get_nodes_from_documents(documents)
    print(f"✓ 生成了 {len(sentence_nodes)} 个句子节点")
    sentence_index_start = time.time()
    sentence_index = VectorStoreIndex.from_documents(documents, transformations=[sentence_parser], show_progress=True)
    sentence_index_time = time.time() - sentence_index_start
    print(f"✓ 句子窗口索引构建完成 ({sentence_index_time:.1f}秒)")

    sentence_query_engine = sentence_index.as_query_engine(
        similarity_top_k=2,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )
    sentence_retriever = sentence_index.as_retriever(similarity_top_k=2)

    # 2.2 常规分块检索（基准）
    print("\n构建常规分块检索...")
    base_parser = SentenceSplitter(chunk_size=512)
    base_nodes = base_parser.get_nodes_from_documents(documents)
    print(f"✓ 生成了 {len(base_nodes)} 个分块节点")
    base_index_start = time.time()
    base_index = VectorStoreIndex.from_documents(documents, transformations=[base_parser], show_progress=True)
    base_index_time = time.time() - base_index_start
    print(f"✓ 常规分块索引构建完成 ({base_index_time:.1f}秒)")

    base_query_engine = base_index.as_query_engine(similarity_top_k=2)
    base_retriever = base_index.as_retriever(similarity_top_k=2)

    # 3. 初始化响应评估器
    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)

    # 4. 执行响应评估对比
    print("开始执行响应评估对比...")
    evaluators = {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator}
    queries = response_eval_dataset.queries

    # 句子窗口检索响应评估
    print("\n=== 评估句子窗口检索 ===")
    sentence_eval_start = time.time()
    sentence_runner = BatchEvalRunner(evaluators, workers=2, show_progress=True)
    sentence_response_results = await sentence_runner.aevaluate_queries(
        queries=queries, query_engine=sentence_query_engine
    )
    sentence_eval_time = time.time() - sentence_eval_start

    # 常规分块检索响应评估
    print("\n=== 评估常规分块检索 ===")
    base_eval_start = time.time()
    base_runner = BatchEvalRunner(evaluators, workers=2, show_progress=True)
    base_response_results = await base_runner.aevaluate_queries(
        queries=queries, query_engine=base_query_engine
    )
    base_eval_time = time.time() - base_eval_start

    # 5. 分析并打印对比结果
    print("\n" + "="*60)
    print("响应评估结果对比")
    print("="*60)

    def calc_response_score(results, metric):
        if results and results.get(metric):
            scores = results[metric]
            return sum(r.passing for r in scores) / len(scores)
        return 0

    # 句子窗口检索结果
    sentence_faith = calc_response_score(sentence_response_results, "faithfulness")
    sentence_rel = calc_response_score(sentence_response_results, "relevancy")

    # 常规分块检索结果
    base_faith = calc_response_score(base_response_results, "faithfulness")
    base_rel = calc_response_score(base_response_results, "relevancy")

    print(f"\n句子窗口检索:")
    print(f"  忠实度: {sentence_faith:.1%}")
    print(f"  相关性: {sentence_rel:.1%}")

    print(f"\n常规分块检索:")
    print(f"  忠实度: {base_faith:.1%}")
    print(f"  相关性: {base_rel:.1%}")



    # 简单对比
    if sentence_faith > base_faith and sentence_rel > base_rel:
        print(f"\n✅ 句子窗口检索在两个维度上都优于常规分块检索")
    elif sentence_faith > base_faith or sentence_rel > base_rel:
        print(f"\n⚖️  句子窗口检索在某些维度上有优势")
    else:
        print(f"\n❌ 句子窗口检索未显示明显优势")

    # 性能统计
    total_time = time.time() - total_start
    print(f"\n{'='*20} 性能统计 {'='*20}")
    print(f"文档加载: {load_time:.1f}秒")
    print(f"句子窗口索引构建: {sentence_index_time:.1f}秒")
    print(f"常规分块索引构建: {base_index_time:.1f}秒")
    print(f"句子窗口评估: {sentence_eval_time:.1f}秒")
    print(f"常规分块评估: {base_eval_time:.1f}秒")
    print(f"总耗时: {total_time:.1f}秒")



if __name__ == "__main__":
    asyncio.run(main())
