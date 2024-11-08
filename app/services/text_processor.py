from .dependencies import List, Document
from . import ProcessorBase

class TextProcessor(ProcessorBase):
    def process(self, data: str, conversation_id: str) -> List[Document]:
        coref_text = self.graph.text_coref(data)
        chunks = self.graph.semantic_chunking(coref_text, 51)
        sub_docs = [Document(text=chunk, metadata={"conversation_id": conversation_id}) for chunk in chunks]
        return sub_docs