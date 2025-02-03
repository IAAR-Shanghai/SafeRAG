from nctd_datasets.xinhua import get_task_datasets
from tasks.nctd_attack import Silver_noise, Inter_context_conflict, Soft_ad, White_DoS
import argparse
from loguru import logger
from llms.api_model import GPT, DeepSeek
from llms.local_model import Qwen_7B_Chat, Qwen_14B_Chat, Baichuan2_13B_Chat, ChatGLM3_6B_Chat
from embeddings.base import HuggingfaceEmbeddings
from retrievers import BaseRetriever, CustomBM25Retriever, EnsembleRetriever, EnsembleRerankRetriever
from evaluator import BaseEvaluator

parser = argparse.ArgumentParser()

#索引设置
parser.add_argument('--clean_docs_path', default='/mnt/data101_d2/simin/simin/SafeRAG/saferag/knowledge_base', help="Path to the retrieval documents")
parser.add_argument('--chunk_size', type=int, default=256, help="Chunk size")
parser.add_argument('--chunk_overlap', type=int, default=0, help="Overlap chunk size")
parser.add_argument('--collection_name', default="chuncksize_256", help="Name of the collection")

#检索器设置
parser.add_argument('--retriever_name', default="bm25", help="Name of the retriever")#''base','bm25','hybrid','hybrid-rerank'
parser.add_argument('--retrieve_top_k', type=int, default=6, help="Top k documents to retrieve")
parser.add_argument('--embedding_name', default='/mnt/data102_d2/huggingface/models/bge-base-zh-v1.5')
parser.add_argument('--embedding_dim', type=int, default=768)

#过滤器设置
parser.add_argument('--filter_module', default='off', help="Name of the filter")#wof/nli/skr

#生成器/评估器设置
# 'gpt-3.5-turbo'、'gpt-4'、'gpt-4o'、'deepseek-chat'、'baichuan2_13b'、'chatglm3_6b'
parser.add_argument('--model_name', default='deepseek-chat', help="Name of the model to use")
parser.add_argument('--quest_eval_model', default='deepseek-chat', help="Name of the model to use")
parser.add_argument('--temperature', type=float, default=0.01, help="Controls the randomness of the model's text generation")
parser.add_argument('--max_new_tokens', type=int, default=4096, help="Maximum number of new tokens to be generated by the model")

#攻击设置
parser.add_argument('--attack_data_path', default='/mnt/data101_d2/simin/simin/SafeRAG/saferag/nctd_datasets/nctd_1015.json', help="Path to the dataset")
parser.add_argument('--attack_task', default='SN', help="Task to perform")
parser.add_argument('--attack_module', default='indexing', help="Phase of attack")#indexing/retrieval/generation
parser.add_argument('--attack_intensity', type=float, default=3/6, help="Proportion of attacks")


#评估设置
parser.add_argument('--shuffle', type=bool, default=True, help="Whether to shuffle the dataset")
parser.add_argument('--bert_score_eval', action='store_true', default=True, help="Whether to use bert_score metrics")
parser.add_argument('--quest_eval', action='store_true', default=True, help="Whether to use MC-QA metrics(Multiple-choice QuestEval)")

#其他设置
parser.add_argument('--num_threads', type=int, default=1, help="Number of threads")
parser.add_argument('--show_progress_bar', action='store', default=True, type=bool, help="Whether to show a progress bar")
parser.add_argument('--contain_original_data', action='store_true', help="Whether to contain original data")

args = parser.parse_args()
logger.info(args)


if args.model_name.startswith("gpt"):
    llm = GPT(model_name=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
elif args.model_name == "deepseek-chat":
    llm = DeepSeek(model_name=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
elif args.model_name == "qwen7b":
    llm = Qwen_7B_Chat(model_name=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
elif args.model_name == "qwen14b":
    llm = Qwen_14B_Chat(model_name=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
elif args.model_name == "baichuan2_13b":
    llm = Baichuan2_13B_Chat(model_name=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
elif args.model_name == "chatglm3_6b":
    llm = ChatGLM3_6B_Chat(model_name=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_tokens)

embed_model = HuggingfaceEmbeddings(model_name=args.embedding_name)

if args.retriever_name == "base":
    retriever = BaseRetriever(
        args.attack_data_path, args.clean_docs_path, args.attack_task, args.attack_module, args.attack_intensity, 
        embed_model=embed_model, embed_dim=args.embedding_dim,
        filter_module = args.filter_module, 
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
        collection_name=args.collection_name, similarity_top_k=args.retrieve_top_k
    )
elif args.retriever_name == "bm25":
    retriever = CustomBM25Retriever(
        args.attack_data_path, args.clean_docs_path, args.attack_task, args.attack_module, args.attack_intensity, 
        embed_model=embed_model, embed_dim=args.embedding_dim,
        filter_module = args.filter_module, 
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
        collection_name=args.collection_name, similarity_top_k=args.retrieve_top_k
    )
elif args.retriever_name == "hybrid":
    retriever = EnsembleRetriever(
        args.attack_data_path, args.clean_docs_path, args.attack_task, args.attack_module, args.attack_intensity, 
        embed_model=embed_model, embed_dim=args.embedding_dim,
        filter_module = args.filter_module, 
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
        collection_name=args.collection_name, similarity_top_k=args.retrieve_top_k
    )
elif args.retriever_name == "hybrid-rerank":
    retriever = EnsembleRerankRetriever(
        args.attack_data_path, args.clean_docs_path, args.attack_task, args.attack_module, args.attack_intensity, 
        embed_model=embed_model, embed_dim=args.embedding_dim,
        filter_module = args.filter_module, 
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
        collection_name=args.collection_name, similarity_top_k=args.retrieve_top_k
    )
else:
    raise ValueError(f"未知检索器: {args.retriever_name}")

task_mapping = {
    'SN':Silver_noise,
    'ICC':Inter_context_conflict,
    'SA': Soft_ad,
    'WDoS': White_DoS
}

if args.attack_task not in task_mapping:
    raise ValueError(f"未知任务: {args.task}")

task = task_mapping[args.attack_task](quest_eval_model=args.quest_eval_model, attack_task=args.attack_task, use_quest_eval=args.quest_eval, use_bert_score=args.bert_score_eval)
dataset = get_task_datasets(args.attack_data_path, args.attack_task)


evaluator = BaseEvaluator(task, llm, retriever, dataset, num_threads=args.num_threads)
evaluator.run(show_progress_bar=args.show_progress_bar, contain_original_data=args.contain_original_data)

