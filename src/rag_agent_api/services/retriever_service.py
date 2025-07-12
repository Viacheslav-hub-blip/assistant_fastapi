from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.rag_agent_api.config import VEC_BASES
from src.rag_agent_api.embeddings_init import embeddings, embedding_function
from src.rag_agent_api.services.database.documents_getter_service import DocumentsGetterService


class CustomRetriever:
    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore

    def get_relevant_documents(self, query: str, belongs_to: Optional[str] = None) -> list[Document]:
        print("===================get docs++++++++++++++++++")
        search_filter = {"belongs_to": belongs_to} if belongs_to else None
        results = self.vectorstore.similarity_search_with_score(query, filter=search_filter)
        collection_name = self.vectorstore._collection.name
        user_id = collection_name.split('_')[1]
        enriched_docs = []
        for doc, score in results:
            doc.metadata["score"] = score
            doc.metadata["source_chunk_content"] = self._get_source_chunk(int(user_id), doc.metadata)
            enriched_docs.append(doc)
        return enriched_docs

    def _get_source_chunk(self, user_id: int, metadata: dict) -> str:
        return DocumentsGetterService.get_source_chunk(
            user_id=user_id,
            workspace_id=metadata["workspace_id"],
            belongs_to=metadata["belongs_to"],
            doc_number=metadata["doc_number"]
        ).page_content


class VectorDBManager:

    @staticmethod
    def get_or_create_retriever(user_id: int, workspace_id: int):
        collection_name = f"user_{user_id}_{workspace_id}"
        client = chromadb.PersistentClient(path=rf"{VEC_BASES}/chroma_db_{user_id}")
        if collection_name in [name for name in client.list_collections()]:
            collection = client.get_collection(collection_name)
        else:
            collection = client.create_collection(collection_name)

        vec_store = Chroma(
            collection_name=collection.name,
            embedding_function=embeddings,
            client=client
        )
        return CustomRetriever(vec_store)

    @staticmethod
    def _copy_collection_to_user(source_user_id: int,
                                 source_collection_name: str,
                                 target_user_id: int,
                                 target_collection_name: str
                                 ) -> bool:
        source_client = chromadb.PersistentClient(path=f"{VEC_BASES}/chroma_db_{source_user_id}")
        target_client = chromadb.PersistentClient(path=f"{VEC_BASES}/chroma_db_{target_user_id}")

        if source_collection_name not in [name for name in source_client.list_collections()]:
            raise ValueError(f"коллекция не найдена у пользователя {source_user_id}")

        source_collection = source_client.get_collection(source_collection_name)
        source_data = source_collection.get()

        target_collection = target_client.get_or_create_collection(
            target_collection_name,
            embedding_function=embedding_function
        )
        target_collection.add(
            ids=source_data["ids"],
            documents=source_data["documents"],
            metadatas=source_data["metadatas"],
            embeddings=source_data["embeddings"]
        )
        return True

    @staticmethod
    def copy_collection(source_user_id: int, source_workspace_id: int, target_user_id: int, target_workspace_id: int):
        return VectorDBManager._copy_collection_to_user(
            source_user_id=source_user_id,
            source_collection_name=f"user_{source_user_id}_{source_workspace_id}",
            target_user_id=target_user_id,
            target_collection_name=f"user_{target_user_id}_{target_workspace_id}"
        )
