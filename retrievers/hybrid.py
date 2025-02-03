from abc import ABC
from operator import itemgetter
from llms.api_model import GPT, DeepSeek
import numpy as np
from llama_index.retrievers import BaseRetriever
from langchain.schema.embeddings import Embeddings

from retrievers import BaseRetriever, CustomBM25Retriever

class EnsembleRetriever(ABC):
    def __init__(
            self, 
            attack_data_directory: str, 
            docs_directory: str, 
            attack_task: str,
            attack_module: str,
            attack_intensity: 0.0,
            embed_model: Embeddings,
            embed_dim: int = 768, 
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
        self.weights = [0.5, 0.5]
        self.c: int = 60
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
        response_text = bm25_response_text + embedding_response_text
        doc_lists = [bm25_response_text, embedding_response_text]

        all_documents = set()
        for doc_list in doc_lists:
            for doc in doc_list:
                all_documents.add(doc)

        rrf_score_dic = {doc: 0.0 for doc in all_documents}
        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score = weight * (1 / (rank + self.c))
                rrf_score_dic[doc] += rrf_score

        sorted_documents = sorted(rrf_score_dic.items(), key=itemgetter(1), reverse=True)
        hybrid_response_text = []
        for sorted_doc in sorted_documents[:self.similarity_top_k]:
            text, score = sorted_doc
            hybrid_response_text.append(text)
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