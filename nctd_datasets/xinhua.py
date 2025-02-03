import json
import os
import random
from typing import Any, Union

from nctd_datasets.base import BaseDataset

class Xinhua(BaseDataset):
    def __init__(self, data, shuffle: bool = False, seed: int = 22):
        self.data = data

        if shuffle:
            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: Union[int, slice]) -> Union[dict, list[dict]]:
            return self.data[key]

    def load(self) -> list[dict]:
        return self.data[:]

    def statistics(self) -> dict:
        stat = {'n': 0, 'c': 0, 't': 0, 'd': 0}
        for type_ in stat.keys():
            stat[type_] = sum([obj['type']==type_ for obj in self.data])
        return stat
    
    def read_output(self, output_path) -> dict:
        with open(output_path, encoding='utf-8') as f:
            return json.load(f)
        


def get_task_datasets(path: str, task: str, shuffle: bool = False, seed: int = 22):
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)        
    return Xinhua(data[task], shuffle, seed).load()[:5]
