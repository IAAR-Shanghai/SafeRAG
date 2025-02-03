from abc import ABC

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.node_parser import SentenceSplitter
from llama_index import ServiceContext, StorageContext
from langchain.schema.embeddings import Embeddings
from llama_index.vector_stores import MilvusVectorStore
import json
from llms.api_model import GPT
from llms.api_model import DeepSeek
import math
import os
import shutil
import numpy as np

class BaseRetriever(ABC):
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
        self.attack_data_directory = attack_data_directory
        self.attack_task = attack_task
        self.docs_directory = docs_directory + '/' + self.attack_task + '/'
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

        if self.attack_module == 'indexing':
            print('选定的攻击模块:', self.attack_module)
            self.construct_attack_index()
        else:
            print('选定的攻击模块:', self.attack_module)
            self.construct_index()

        retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=self.similarity_top_k,
        )
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
        )

    def construct_index(self):
        storage_context, service_context = self.initialize_storage_and_service_context(overwrite=True)
        documents = SimpleDirectoryReader(self.docs_directory).load_data()
        splitter = SentenceSplitter.from_defaults(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap, 
            secondary_chunking_regex="[^\n。]+[\n。]?"
        )
        nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
        for spilt_ids in range(0, len(nodes), 8000):
            self.vector_index = GPTVectorStoreIndex(
                nodes[spilt_ids:spilt_ids+8000], 
                service_context=service_context, 
                storage_context=storage_context, 
                show_progress=True
            )
            print(f"Indexing of part {spilt_ids} finished!")
        print("Indexing finished!")

    def construct_attack_index(self):
        self.prepare_attack_documents()
        storage_context, service_context = self.initialize_storage_and_service_context(
            collection_name=self.collection_name_attack, 
            overwrite=True 
        )
        documents = SimpleDirectoryReader(self.attack_docs_directory).load_data()
        splitter = SentenceSplitter.from_defaults(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap, 
            secondary_chunking_regex="[^\n。]+[\n。]?"
        )
        nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
        for spilt_ids in range(0, len(nodes), 8000):
            self.vector_index = GPTVectorStoreIndex(
                nodes[spilt_ids:spilt_ids+8000], 
                service_context=service_context, 
                storage_context=storage_context, 
                show_progress=True
            )
            print(f"Indexing of part {spilt_ids} finished!")
        print("Indexing finished!")

    def prepare_attack_documents(self):
        if self.attack_task in ['SN', 
                                'ICC',  
                                'SA', 
                                'WDoS'
                                ]:
            print(f'添加 {self.attack_intensity} {self.attack_task} 到 {self.docs_directory} 目录中...')
            self.attack_docs_directory = self.docs_directory + 'add_' + self.attack_task
            if os.path.exists(self.attack_docs_directory):
                shutil.rmtree(self.attack_docs_directory)
            os.makedirs(self.attack_docs_directory)
            docs_path = os.path.join(self.docs_directory, 'db.txt')
            self.attack_docs_path = os.path.join(self.attack_docs_directory, 'db.txt')
            shutil.copy2(docs_path, self.attack_docs_path)
            attack_data_points = self.read_safe_rag_data()[self.attack_task]
            for attack_data_point in attack_data_points:
                noise_contexts = attack_data_point['enhanced_'+ self.attack_task +'_contexts']
                num_to_select = math.ceil(self.similarity_top_k * self.attack_intensity)
                selected_attack_contexts = noise_contexts[:num_to_select]
                for attack_context in selected_attack_contexts:
                    self.save_attack_db(attack_context)

    def initialize_storage_and_service_context(self, overwrite=False, collection_name=None):
        if collection_name is None:
            collection_name = self.collection_name
        vector_store = MilvusVectorStore(
            dim=self.embed_dim, 
            overwrite=overwrite, 
            collection_name=collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        service_context = ServiceContext.from_defaults(embed_model=self.embed_model, llm=None)
        return storage_context, service_context

    def read_safe_rag_data(self) -> dict:
        with open(self.attack_data_directory, encoding='utf-8') as f:
            return json.load(f)
 
    def save_attack_db(self, output: str) -> None:
        with open(self.attack_docs_path, 'a', encoding='utf-8') as f:
            f.write(output + '\n')

    def search_docs(self, query_text: str):
        response_vector = self.query_engine.query(query_text)
        response_text_list = response_vector.response.split('\n---------------------\n')
        response_text = response_text_list[1].split("\n\n")
        response_text = [text for text in response_text if not text.startswith("file_path: ")]

        if self.attack_module == 'retrieval' or self.attack_module == 'generation':
            attack_data_points = self.read_safe_rag_data()[self.attack_task]
            for attack_data_point in attack_data_points:
                if attack_data_point['questions'] == query_text:
                    attack_contexts = attack_data_point['enhanced_'+self.attack_task+'_contexts']
                    num_to_select = math.ceil(self.similarity_top_k * self.attack_intensity)
                    selected_attack_contexts = attack_contexts[:num_to_select]
            response_text = selected_attack_contexts + response_text
            response_text = response_text[:self.similarity_top_k]
        filtered_response_text = "\n\n".join(response_text)

        if self.filter_module == 'off':
            print('不使用过滤器')
        elif self.filter_module == 'skr':
            filtered_response_text = self.filter(query_text, (response_text), self.filter_module)
            filtered_response_text = "\n\n".join(filtered_response_text)  
        elif self.filter_module == 'nli':
            filtered_response_text = self.filter(query_text, (response_text), self.filter_module)
            filtered_response_text = "\n\n".join(filtered_response_text)  
        return response_text, filtered_response_text
    
    def filter(self, query_text: str, response_text:list, filter_module:str):
        filtered_response_text = self.ds.filter(query_text, str(response_text), filter_module)
        return filtered_response_text



