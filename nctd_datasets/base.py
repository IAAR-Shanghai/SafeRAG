from abc import ABC, abstractmethod
from typing import Union


class BaseDataset(ABC):
    @abstractmethod
    def __init__(self, path):
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, key: Union[int, slice]) -> Union[dict, list[dict]]:
        ...

    @abstractmethod
    def load(self) -> list[dict]:
        ...
