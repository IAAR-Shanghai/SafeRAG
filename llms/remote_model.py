import requests
import json
from loguru import logger

from llms.base import BaseLLM
from importlib import import_module


conf = import_module("configs.config")


class Baichuan2_13B_Chat(BaseLLM):
    def request(self, query) -> str:
        url = conf.Baichuan2_13B_url
        payload = json.dumps({
            "prompt": query,
            "params": {
                "temperature": self.params['temperature'],
                "do_sample": True,
                "max_new_tokens": self.params['max_new_tokens'],
                "num_return_sequences": 1,
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        headers = {
        'token': conf.Baichuan2_13B_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['choices'][0]
        return res


class ChatGLM2_6B_Chat(BaseLLM):
    def request(self, query) -> str:
        url = conf.ChatGLM2_url
        payload = json.dumps({
            "prompt": query,
            "params": {
                "temperature": self.params['temperature'],
                "do_sample": True,
                "max_new_tokens": self.params['max_new_tokens'],
                "num_return_sequences": 1,
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        headers = {
        'token': conf.ChatGLM2_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['choices'][0]
        return res


class Qwen_14B_Chat(BaseLLM):
    def request(self, query) -> str:
        url = conf.Qwen_url
        payload = json.dumps({
            "prompt": query,
            "params": {
                "temperature": self.params['temperature'],
                "do_sample": True,
                "max_new_tokens": self.params['max_new_tokens'],
                "num_return_sequences": 1,
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        headers = {
        'token': conf.Qwen_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['choices'][0]
        return res

