import copy
import json
import os
from abc import ABC
from loguru import logger
from tqdm import tqdm
from threading import Lock
from llms.base import BaseLLM
from tasks.base import BaseTask
from retrievers.base import BaseRetriever
import concurrent.futures

class BaseEvaluator(ABC):
    def __init__(self, task: BaseTask, model: BaseLLM, retriever: BaseRetriever,
        dataset: list[dict], output_dir: str = './output/', num_threads: int = 40):

        self.model = model
        self.retriever = retriever
        self.dataset = dataset
        self.task = task
        self.lock = Lock()
        self.num_threads = num_threads
        model_name = self.model.model_name
        quest_eval_model = self.task.quest_eval_model
        attack_module = self.retriever.attack_module
        attack_intensity = self.retriever.attack_intensity
        similarity_top_k = self.retriever.similarity_top_k
        filter_module = self.retriever.filter_module
        output_dir = os.path.join(output_dir, f'{retriever.__class__.__name__}_{filter_module}_{model_name}_{quest_eval_model}_{attack_module}_{task.__class__.__name__}')
        if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
            os.makedirs(output_dir)
        self.output_path = os.path.join(
            output_dir, f'{attack_intensity*100}%_top{similarity_top_k}.json'
        )
        self.task.set_model(self.model, self.retriever)

    def task_generation(self, data_point):
        try:
            self.lock.acquire()
            retrieve_context, filtered_response_text = self.task.retrieve_docs(data_point)
            self.lock.release()
            data_point["retrieve_context"] = str(retrieve_context)
            data_point["filtered_retrieve_context"] = filtered_response_text
        except Exception as e:
            logger.warning(repr(e))
            self.lock.release()
            data_point["retrieve_context"] = ''
        return self.task.model_generation(data_point)

    def multithread_batch_scoring(self, dataset: list[dict], sort=True, show_progress_bar=False, contain_original_data=False) -> list[dict]:
        if os.path.exists(self.output_path): 
            results = self.read_output().get('results', [])
            results = self.remove_invalid(results)
            saved_ids = [result['id'] for result in results]
        else:
            results = []
            saved_ids = []
        def process_data_point(data_point):
            print('data_point_id:', data_point['id'])
            if data_point['id'] in saved_ids:
                return None  
            try:
                generated_text = self.task_generation(data_point)
                if generated_text == '","msg":"request openai failed"':
                    return None
                
                data_point["generated_text"] = generated_text
                print('生成的答案:', generated_text)
                result = {'id': data_point['id'], **self.task.scoring(data_point)}
                
                if contain_original_data:
                    result['original_data'] = data_point
                return result
            except Exception as e:
                logger.warning(repr(e))
                return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_results = list(tqdm(executor.map(process_data_point, dataset), total=len(dataset)))
        results.extend([result for result in future_results if result is not None])
        return sorted(results, key=lambda x: x['id']) if sort else results

    def save_output(self, output: dict) -> None:
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    
    def read_output(self) -> dict:
        with open(self.output_path) as f:
            return json.load(f)

    def run(self, sort = True, show_progress_bar = False, contain_original_data = True) -> dict:
        info = {
            'task': self.task.__class__.__name__, 
            'llm': str(self.model.params),
        }
        results = self.multithread_batch_scoring(self.dataset, sort, show_progress_bar, contain_original_data)
        valid_results = self.remove_invalid(results)
        try:
            overall = self.task.compute_overall(valid_results) if len(valid_results) > 0 else {}
        except Exception as e:
            logger.warning(repr(e))
            overall = dict()
        self.save_output(output:={'info': info, 'overall': overall, 'results': results})
        print(f'Output saved at {self.output_path}!')
        return output

    @staticmethod
    def remove_invalid(results: list[dict]) -> list[dict]:
        return [result for result in results if result['valid']]

    def batch_scoring(self, dataset:list[dict], sort = True, show_progress_bar = False, contain_original_data = False):
        if os.path.exists(self.output_path): 
            results = self.read_output().get('results', [])
            results = self.remove_invalid(results)
            saved_ids = [result['id'] for result in results]
        else:
            results = []
            saved_ids = []
        for data_point in (tqdm(dataset, desc=self.model.params['model_name']) if show_progress_bar else dataset):
            if data_point['id'] in saved_ids:
                continue 
            try:
                generated_text = self.task_generation(data_point)
                data_point["generated_text"] = generated_text
                result = {'id': data_point['id'], **self.task.scoring(data_point)}
                if contain_original_data:
                    result['original_data'] = data_point
                results.append(result)
            except Exception as e:
                logger.warning(repr(e))
        return sorted(results, key=lambda x: x['id']) if sort else results
