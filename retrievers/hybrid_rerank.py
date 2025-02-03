from abc import ABC
from typing import List
from llms.api_model import GPT, DeepSeek
import numpy as np
from llama_index.retrievers import BaseRetriever
from langchain.schema.embeddings import Embeddings
from retrievers import BaseRetriever, CustomBM25Retriever
from FlagEmbedding import FlagReranker

def bge_rerank_result(query_text: str, docs: List[str], top_n):
    reranker = FlagReranker('/mnt/data102_d2/huggingface/models/bge-reranker-base')
    scores = reranker.compute_score([[query_text, passage] for passage in docs])
    score_doc_pairs = zip(scores, docs)
    sorted_pairs = sorted(score_doc_pairs, key=lambda x: x[0], reverse=True)
    sorted_docs = [doc for _, doc in sorted_pairs]
    return sorted_docs[:top_n]

class EnsembleRerankRetriever(ABC):
    def __init__(
            self, 
            attack_data_directory: str, 
            docs_directory: str, 
            attack_task: str,
            attack_module: str,
            attack_intensity: 0.0,
            embed_model: Embeddings,
            embed_dim: int = 768, #768,
            filter_module: str = 'base',
            chunk_size: int = 128,
            chunk_overlap: int = 0,
            collection_name: str = "docs",
            similarity_top_k: int=2,
        ):
        super().__init__()
        self.attack_data_directory = attack_data_directory
        self.docs_directory = docs_directory
        self.attack_task = attack_task
        self.attack_module = attack_module
        self.attack_intensity = attack_intensity
        self.filter_module = filter_module
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.collection_name_attack = self.collection_name + '_attack'
        self.similarity_top_k = similarity_top_k
        self.vector_index = None
        self.gpt = GPT(model_name='gpt-3.5-turbo', report=True)
        self.ds = DeepSeek(model_name='deepseek-chat', report=True)

        self.embedding_retriever = BaseRetriever(
        attack_data_directory, docs_directory, attack_task, attack_module, attack_intensity, 
        embed_model=embed_model, embed_dim=embed_dim,
        filter_module = filter_module, 
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        collection_name=collection_name, similarity_top_k=similarity_top_k
    )
        self.bm25_retriever = CustomBM25Retriever(
        attack_data_directory, docs_directory, attack_task, attack_module, attack_intensity, 
        embed_model=embed_model, embed_dim=embed_dim,
        filter_module = filter_module, 
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        collection_name=collection_name, similarity_top_k=similarity_top_k
    )
    

    def search_docs(self, query_text: str):
        bm25_response_text, _ = self.bm25_retriever.search_docs(query_text)
        embedding_response_text, _ = self.embedding_retriever.search_docs(query_text)
        
        doc_lists = [bm25_response_text, embedding_response_text]
        response_text = bm25_response_text + embedding_response_text
        all_documents = set()
        for doc_list in doc_lists:
            for doc in doc_list:
                all_documents.add(doc)

        doc_lists = list(all_documents)
        hybrid_response_text = bge_rerank_result(query_text, doc_lists, top_n=self.similarity_top_k)
        filtered_response_text = "\n\n".join(hybrid_response_text)
        if self.filter_module == 'off':
            print('不使用过滤器')
        elif self.filter_module == 'skr':
            filtered_response_text = self.filter(query_text, (hybrid_response_text), self.filter_module)
            filtered_response_text = "\n\n".join(filtered_response_text)  
        elif self.filter_module == 'nli':
            filtered_response_text = self.filter(query_text, (hybrid_response_text), self.filter_module)
            filtered_response_text = "\n\n".join(filtered_response_text)    
        return response_text, filtered_response_text

    def filter(self, query_text: str, response_text:list, filter_module:str):
        filtered_response_text = self.gpt.filter(query_text, str(response_text), filter_module)
        return filtered_response_text