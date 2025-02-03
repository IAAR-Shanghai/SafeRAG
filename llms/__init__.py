from importlib import import_module
conf = import_module("configs.config")

if conf.GPT_api_key != '':
    from .api_model import GPT
elif conf.GPT_transit_url != '':
    from .remote_model import GPT

from .local_model import Qwen_7B_Chat, Qwen_14B_Chat, Baichuan2_13B_Chat, ChatGLM3_6B_Chat
