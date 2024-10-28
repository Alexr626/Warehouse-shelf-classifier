from abc import ABC, abstractmethod



class BaseModel(ABC):
    type: str
    def __init__(self, type):
        super().__init__()
        self.type = type

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        raise NotImplementedError





