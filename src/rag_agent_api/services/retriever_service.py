import chromadb
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.rag_agent_api.embeddings_init import embeddings
from src.rag_agent_api.services.database.documents_getter_service import DocumentsGetterService
from src.rag_agent_api.config import VEC_BASES


class CustomRetriever:
    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore

    def get_relevant_documents(self, workspace_id: int, query: str, belongs_to: str = None) -> list[Document]:
        if belongs_to:
            result_search_sim_docs = self.vectorstore.similarity_search_with_score(query, k=10,
                                                                                   filter={
                                                                                       "workspace_id": workspace_id,
                                                                                       "belongs_to": belongs_to
                                                                                   })
        else:
            result_search_sim_docs = self.vectorstore.similarity_search_with_score(query)
        collection_name = self.vectorstore._collection_name
        user_id = collection_name.split('_')[1]
        result = []
        for result_search_sim_doc, score in result_search_sim_docs:
            belongs_to = result_search_sim_doc.metadata["belongs_to"]
            doc_number = result_search_sim_doc.metadata["doc_number"]
            result_search_sim_doc.metadata["score"] = score
            source_chunk = DocumentsGetterService.get_source_chunk(user_id, workspace_id, belongs_to, doc_number)
            result_search_sim_doc.metadata["source_chunk_content"] = source_chunk.page_content
            result.append(result_search_sim_doc)
        return result


class RetrieverSrvice:

    @staticmethod
    def get_or_create_retriever(user_id: int):
        """Создает векторноую базу и retriever для пользователя, если она не была найдена
        Если такое хранилище существует, возвращает существующие хранилище
        """
        collection_name = f"user_{user_id}"
        client = chromadb.PersistentClient(path=rf"{VEC_BASES}\chroma_db_{user_id}")
        if collection_name in [name for name in client.list_collections()]:
            collection = client.get_collection(collection_name)
            vec_store = Chroma(
                collection_name=collection.name,
                embedding_function=embeddings,
                client=client,
                collection_metadata={"hnsw:space": "cosine"},

            )
            retriever = CustomRetriever(
                vectorstore=vec_store,
            )
            return retriever

        vec_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=rf"{VEC_BASES}\chroma_db_{user_id}",
            collection_metadata={"hnsw:space": "cosine"}
        )
        retriever = CustomRetriever(
            vectorstore=vec_store,
        )
        return retriever
