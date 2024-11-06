from .dependencies import ABC, Graph, abstractmethod, List, Document

class ProcessorBase(ABC):
    def __init__(self):
        self.graph = Graph()

    @abstractmethod
    def process(self, data, conversation_id: int) -> List[Document]:
        pass

    def add_to_graph(self, data, conversation_id: int):
        sub_docs = self.process(data, conversation_id)
        self.graph.add_to_graph(sub_docs)