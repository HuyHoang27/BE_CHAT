from . import TextProcessor, ImageProcessor
from .dependencies import Graph, List
import nest_asyncio
nest_asyncio.apply()
class LLMService:
    def __init__(self):
        self.graph = Graph()
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        # self.pdf_processor = PDFProcessor()

    def add_to_graph(
        self,
        conversation_id: int,
        text_docs: str = None,
        image_docs: List[bytes] = None,
        pdf_docs: List[bytes] = None,
    ):
        sub_docs = []

        try:
            if text_docs is not None:
                processed_texts = self.text_processor.process(text_docs, conversation_id)
                sub_docs.extend(processed_texts)

            if image_docs is not None:
                processed_images = self.image_processor.process(image_docs, conversation_id)
                sub_docs.extend(processed_images)

            # Process PDF documents if provided
            # if pdf_docs is not None:
            #     processed_pdfs = [await self.pdf_processor.process(doc) for doc in pdf_docs]
            #     sub_docs.extend(processed_pdfs)

            print("LLMService Ok")
            self.graph.add_to_graph(sub_docs)

        except Exception as e:
            print(f"An error occurred while adding to the graph: {e}")
            raise

    def query(self, conversation_id: str, query: str) -> str:
        return self.graph.query(conversation_id, query)
