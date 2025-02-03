import copy
from abc import ABC, abstractmethod
import os
import json
from loguru import logger

class BaseLLM(ABC):
    def __init__(
            self, 
            model_name: str = "gpt-3.5-turbo", 
            temperature: float = 1.0, 
            max_new_tokens: int = 1024, 
            top_p: float = 0.9,
            top_k: int = 5,
            **more_params
        ):
        self.params = {
            'model_name': model_name if model_name else self.__class__.__name__,
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'top_k': top_k,
            **more_params
        }

    def update_params(self, inplace: bool = True, **params):
        if inplace:
            self.params.update(params)
            return self
        else:
            new_obj = copy.deepcopy(self)
            new_obj.params.update(params)
            return new_obj

    @abstractmethod
    def request(self, query:str) -> str:
        return ''

    def safe_request(self, query: str) -> str:
        """Safely make a request to the language model, handling exceptions."""
        try:
            print("input-------------------------:\n", query)
            response = self.request(query)
            print('output------------------------:\n', response)
        except Exception as e:
            logger.warning(repr(e))
            response = ''
        return response

    def filter(self, questions:str, contexts:str, filter_module:str):
        if filter_module == 'nli':
            prompt_file = 'filter/nli.txt'
        elif filter_module == 'skr':
            prompt_file = 'filter/skr.txt'
        template = self._read_prompt_template(prompt_file)
        query = template.format(questions=questions, contexts=contexts)
        res = self.safe_request(query)
        filtered_contexts = res.split('<response>')[-1].split('</response>')[0].strip('\n').split('\n')
        print('过滤上下文为：', filtered_contexts)
        return filtered_contexts


    @staticmethod
    def _read_prompt_template(filename: str) -> str:
        path = os.path.join('prompts/', filename)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return f.read()
        else:
            logger.error(f'Prompt template not found at {path}')
            return ''
