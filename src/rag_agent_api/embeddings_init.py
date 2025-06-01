import os
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from src.rag_agent_api.config import embeddings_model_name, HF_TOKEN

os.environ['HF_TOKEN'] = HF_TOKEN

embeddings = HuggingFaceEmbeddings(
    model_name=embeddings_model_name
)


class ChromaCompatibleEmbeddingFunction:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(input, convert_to_numpy=True).tolist()


embedding_function = ChromaCompatibleEmbeddingFunction(model_name=embeddings_model_name)
