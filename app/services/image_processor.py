from . import ProcessorBase
from .dependencies import (
    List,
    Document,
    PdfMerger,
    os,
    Image,
    BytesIO,
    pytesseract,
    LlamaParse,
    TextNode,
    deepcopy
)

class ImageProcessor(ProcessorBase):
    async def process(self, data: List[bytes], conversation_id: int) -> List[Document]:
        sub_docs = await self.process_images(data, conversation_id)
        return sub_docs

    async def process_images(self, data: List[bytes], conversation_id: int) -> List[Document]:
        temp_dir = './temp_pdfs'
        os.makedirs(temp_dir, exist_ok=True)

        merger = PdfMerger()
        temp_pdf_files = []

        for idx, image_file in enumerate(data):
            # Open the image
            image = Image.open(BytesIO(image_file))

            # Convert the image to a searchable PDF (binary data)
            pdf_bytes = pytesseract.image_to_pdf_or_hocr(image, extension='pdf')

            # Generate a temporary file name for each image
            temp_pdf_path = os.path.join(temp_dir, f'temp_{idx}.pdf')
            with open(temp_pdf_path, 'wb') as f:
                f.write(pdf_bytes)

            # Append the temp PDF file to the merger
            merger.append(temp_pdf_path)

            # Add the temp file to the list for later deletion
            temp_pdf_files.append(temp_pdf_path)

        output_pdf_path = './merged_output.pdf'
        with open(output_pdf_path, 'wb') as output_pdf:
            merger.write(output_pdf)

        # Close the merger
        merger.close()

        # Delete the temporary PDF files
        for temp_pdf in temp_pdf_files:
            os.remove(temp_pdf)

        documents = LlamaParse(result_type="markdown").load_data(output_pdf_path)
        page_nodes = self.get_page_nodes(documents)

        text = ''
        for node in page_nodes:
            content = node.get_content()
            text += content + '\n\n'

        os.remove(output_pdf_path)
        
        coref_text = self.graph.text_coref(text)
        chunks = self.graph.semantic_chunking(coref_text, 51)
        sub_docs = [Document(text=chunk, metadata={"doc_id": conversation_id}) for chunk in chunks]
        return sub_docs

    def get_page_nodes(self, docs, separator="\n---\n"):
        nodes = []
        for doc in docs:
            doc_chunks = doc.text.split(separator)
            for doc_chunk in doc_chunks:
                node = TextNode(
                    text=doc_chunk,
                    metadata=deepcopy(doc.metadata),
                )
                nodes.append(node)

        return nodes