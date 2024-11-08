from .dependencies import ABC, Graph, abstractmethod, List, Document

class ProcessorBase(ABC):
    def __init__(self):
        self.graph = Graph()

    @abstractmethod
    def process(self, data, conversation_id: str) -> List[Document]:
        pass
