import json
import jieba
jieba.initialize() 
from loguru import logger
from llms.api_model import DeepSeek
from importlib import import_module

try:
    conf = import_module("configs.config")
except ImportError:
    conf = import_module("configs.real_config")

class QuestEval(DeepSeek):
    def __init__(self, model_name='deepseek-chat', temperature=0.01, max_new_tokens=4096, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report
        self.model_name = model_name
        print('选定的评估器:', self.model_name)
        
    def get_correct_or_incorrect_options(self, answers:str, numbered_options:list):
        prompt_file = 'eval/multiple_choice_eval.txt'
        template = self._read_prompt_template(prompt_file)
        query = template.format(answers=answers, numbered_options=str(numbered_options))
        res = self.safe_request(query)
        try:
            if '```json' in res:
                real_content = res.replace('```json', '').replace('```', '').strip()
                eval_result = json.loads(real_content)
            else:
                eval_result = json.loads(res)
            reason = eval_result['reason']
            correct_options = eval_result['correct_options']
            incorrect_options = eval_result['incorrect_options']
        except json.JSONDecodeError as e:
            print(f'JSON解析错误', e)
        except:
            print(f'获取选项失败')
        return reason, correct_options, incorrect_options

    def mc_eval(self, data_point: dict):
        try:
            reason, selected_correct_options, selected_incorrect_options = self.get_correct_or_incorrect_options(data_point['generated_text'], data_point['numbered_options'])
            ground_truth_correct_options = data_point['ground_truth_correct_options']
            ground_truth_incorrect_options = data_point['ground_truth_incorrect_options']
            print('ground_truth_correct_options:', ground_truth_correct_options)
            print('ground_truth_incorrect_options:', ground_truth_incorrect_options) 
            print('selected_correct_options:', selected_correct_options)
            print('selected_incorrect_options:', selected_incorrect_options) 
            print('reason:', reason)          

            quest_eval_save = {}
            quest_eval_save["selected_correct_options"] = selected_correct_options
            quest_eval_save["selected_incorrect_options"] = selected_incorrect_options
            quest_eval_save["reason"] = reason
            f1_correct = compute_f1(selected_correct_options, ground_truth_correct_options)
            f1_incorrect = compute_f1(selected_incorrect_options, ground_truth_incorrect_options)
            return f1_correct, f1_incorrect, quest_eval_save
        except Exception as e:
            logger.warning(repr(e))
            quest_eval_save = {}
            quest_eval_save["correct_options"] = []
            quest_eval_save["incorrect_options"] = []
            quest_eval_save["reason"] = []
            return -1, -1, quest_eval_save

def compute_f1(selected_options, ground_truth_options):
    TP = len(set(selected_options).intersection(ground_truth_options))
    FP = len(set(selected_options) - set(ground_truth_options))
    FN = len(set(ground_truth_options) - set(selected_options))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1
