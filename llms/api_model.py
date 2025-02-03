import openai
from loguru import logger
import json
import requests
from llms.base import BaseLLM
from importlib import import_module

conf = import_module("configs.config")

class GPT(BaseLLM):
    def __init__(self, model_name='gpt-3.5-turbo', temperature=0.01, max_new_tokens=4096, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report
        self.model_name = model_name
        self.temperature = temperature
    def request(self, query: str) -> str:
        url = conf.GPT_api_base
        headers = {
            "Content-Type": "application/json",
            "Authorization": conf.GPT_api_key
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", 
                        "content": query
                        }],
            "temperature": self.temperature
        }
        json_data = json.dumps(data)
        res = requests.post(url, headers=headers, data=json_data)
        res = res.json()
        real_res = res["choices"][0]["message"]["content"]
        token_consumed = res['usage']['total_tokens']
        logger.info(f'GPT token consumed: {token_consumed}') #if self.report else ()
        return real_res

class DeepSeek(BaseLLM):
    def __init__(self, model_name='deepseek-chat', temperature=0.01, max_new_tokens=4096, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report
        self.model_name = model_name
    def request(self, query: str) -> str:
        deepseek = openai.OpenAI(api_key=conf.DeepSeek_key, base_url=conf.DeepSeek_base)
        res = deepseek.chat.completions.create(
            model= self.params['model_name'],
            messages=[{"role": "system", "content": query},],
            temperature = self.params['temperature'],
            max_tokens = self.params['max_new_tokens'],
            top_p = self.params['top_p'],
            stream=False
        )
        real_res = res.choices[0].message.content
        token_consumed = res.usage.total_tokens
        logger.info(f'GPT token consumed: {token_consumed}') if self.report else ()
        return real_res
    