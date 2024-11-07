import os
from dotenv import load_dotenv
from typing import List, Optional, Type
from pydantic import BaseModel, Field
# Import libraries for NLP and LLM processing
import spacy
from fastcoref import spacy_component
from llama_index.llms.groq import Groq
from llama_index.core import QueryBundle
# Additional imports
import nltk
# Uncomment the line below if you want to use HuggingFaceLLM from llama_index
# from llama_index.llms.huggingface import HuggingFaceLLM

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.schema import Document
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import (
    DynamicLLMPathExtractor,
    LLMSynonymRetriever,
    VectorContextRetriever,
    CypherTemplateRetriever,
)
import os
# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
LLAMA_CLOUD_API_KEY = os.environ["LLAMA_CLOUD_API_KEY"]

import nest_asyncio

nest_asyncio.apply()

class CustomGraph:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CustomGraph, cls).__new__(cls)
        return cls._instance

    def __init__(self, llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct", embed_model: str = "BAAI/bge-base-en-v1.5"):
        if not hasattr(self, 'initialized'):  # Avoid reinitializing on repeated instantiation
            self.llm, self.embed_model = self.load_model(llm_model, embed_model)
            self.nlp = self.load_spacy_pipeline()
            self.graph_store = self.load_neo4j_graph_store()
            self.check_nltk_downloaded()
            self.kg_extractor = self.initialize_kg_extractor()
            self.index = self.initialize_index()
            self.llm_synonym, self.vector_context = self.initialize_retrievers()
            self.query_engine = self.initialize_query_engine()
            self.initialized = True

    @staticmethod
    def check_nltk_downloaded():
        nltk.download('punkt_tab', quiet=True)

    @staticmethod
    def load_spacy_pipeline():
        try:
            nlp = spacy.load("en_core_web_sm")
            nlp.add_pipe("fastcoref")
            return nlp
        except Exception as e:
            print(f"Error loading spaCy model or adding coref pipeline: {e}")
            return None

    @staticmethod
    def load_model(llm_model: str, embed_model: str):
        try:
            # llm_md = HuggingFaceLLM(
            #     context_window=4096,
            #     max_new_tokens=2048,
            #     generate_kwargs={"temperature": 0.1, "top_k": 1},
            #     tokenizer_name=llm_model,
            #     model_name=llm_model,
            #     device_map="auto",
            #     model_kwargs={
            #         "token": os.getenv("HF_TOKEN"),
            #         # "disk_offload": True,
            #         },
            #     tokenizer_kwargs={"token": os.getenv("HF_TOKEN")},
            # )
            llm_md = Groq(model="llama3-8b-8192", api_key=os.getenv('GROQ_API_KEY'))
            embed_md = HuggingFaceEmbedding(model_name=embed_model)
            Settings.llm = llm_md
            Settings.embed_model = embed_md
            return llm_md, embed_md
        except Exception as e:
            print(f"Error initializing models: {e}")
            return None, None

    def initialize_kg_extractor(self):
        return [
            DynamicLLMPathExtractor(
                llm=self.llm,
                max_triplets_per_chunk=10,
                num_workers=20,
            )
        ]

    def initialize_index(self):
        return PropertyGraphIndex.from_existing(
            property_graph_store=self.graph_store,
            embed_model=self.embed_model,
            kg_extractors=self.kg_extractor,
            show_progress=True,
        )

    def initialize_retrievers(self):
        
        llm_synonym = LLMSynonymRetriever(
            self.index.property_graph_store,
            llm=self.llm,
            include_text=False,
        )
        vector_context = VectorContextRetriever(
            self.index.property_graph_store,
            embed_model=self.embed_model,
            include_text=False,
        )
        return llm_synonym, vector_context

    def initialize_query_engine(self):
        return self.index.as_query_engine(
            sub_retrievers=[self.llm_synonym, self.vector_context],
            llm=self.llm
        )

    @staticmethod
    def load_neo4j_graph_store() -> Optional[Neo4jPGStore]:
        try:
            username = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")
            url = os.getenv("NEO4J_URL")
            if not all([username, password, url]):
                raise ValueError("Missing required environment variables for Neo4j configuration.")

            return Neo4jPGStore(username=username, password=password, url=url)
        except Exception as e:
            print(f"Error creating Neo4j graph store: {e}")
            return None

    def text_coref(self, input_text: str) -> str:
        try:
            doc = self.nlp(input_text, component_cfg={"fastcoref": {'resolve_text': True}})
            return doc._.resolved_text
        except Exception as e:
            print(f"Error processing text coreference: {e}")
            return input_text

    @staticmethod
    def semantic_chunking(input_text: str, threshold_value: int) -> List[str]:
        try:
            doc = nltk.sent_tokenize(input_text)
            chunks = []
            current_chunk = []

            for sent in doc:
                current_chunk.append(sent)
                if len(" ".join(current_chunk)) > threshold_value:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            return chunks
        except Exception as e:
            print(f"Error in semantic chunking: {e}")
            return []

    def add_to_graph(self, sub_docs: List[Document]):
        PropertyGraphIndex.from_documents(
            sub_docs,
            embed_model=self.embed_model,
            kg_extractors=self.kg_extractor,
            property_graph_store=self.graph_store,
            show_progress=True,
        )

    def query(self, conversation_id: str, query: str) -> str:
        new_query = """
        {query}
        Based on the Cypher queries you've retrieved, focus only on nodes where conversation_id = {conversation_id}. 
        For the selected nodes, please provide the following details in the specified format:
        Result:
        <Answer here>
        Explain:
        <Answer here>

        Note: If there are no nodes with conversation_id = {conversation_id}, respond with:
        'I cannot answer your question. Please provide more context.'

        Output in Result should be in one of the following formats:
        - If you have an answer: 'The correct answer is (A/B/C/D)'
        - If you cannot answer: 'I cannot answer your question. Please provide more context.'
        """.format(conversation_id=conversation_id, query=query)



        # Xử lý dữ liệu nhận được từ truy vấn
        query_doc = self.index.as_query_engine(
            sub_retrievers=[self.llm_synonym, self.vector_context],
            llm=self.llm
        ).query(new_query)
        
        return str(query_doc)
