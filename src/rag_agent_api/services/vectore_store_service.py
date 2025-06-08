import re
from typing import List, NamedTuple
from langchain.schema.document import Document
import chromadb
from src.rag_agent_api.services.retriever_service import CustomRetriever
from src.rag_agent_api.services.llm_model_service import LLMModelService, SummarizeContentAndDocs
from src.rag_agent_api.services.database.documents_saver_service import DocumentsSaverService

from src.rag_agent_api.services.text_splitter_service import TextSplitterService
from src.rag_agent_api.config import VEC_BASES


class SummDocsWithSourceAndIds(NamedTuple):
    summarize_docs_with_ids: List[Document]
    doc_ids: List[str]
    source_docs: List[Document]


class VecStoreService:
    def __init__(self,
                 model_service: LLMModelService,
                 retriever: CustomRetriever,
                 content: str,
                 file_name: str,
                 user_id: int,
                 work_space_id: int
                 ) -> None:
        self.model_service = model_service
        self.retriever = retriever
        self.content = content
        self.file_name = file_name
        self.user_id = user_id
        self.work_space_id = work_space_id

    def _get_summary_doc_content(self, split_docs: List[str]) -> SummarizeContentAndDocs:
        """Создает сжатые документы из полных фрагментов
        Если документ всег один(его длина была слшком маленькой для разделения,он остается без изменений)
        Иначе получаем SummarizeContentAndDocs с сжатыми документами и исходными
        """
        if len(split_docs) == 1:
            return SummarizeContentAndDocs(split_docs, split_docs)
        return self.model_service.get_summarize_docs_with_questions(split_docs)

    def get_chunks(self) -> list[str]:
        source_split_documents: list[str] = TextSplitterService.get_semantic_split_documents(self.content)
        return source_split_documents

    def get_summarize_chunks(self, chunks: list[str]) -> list[str]:
        summarized_docs: list[str] = [sum for sum in
                                      self._get_summary_doc_content(chunks).summary_texts]
        return summarized_docs

    def add_metadata_to_chunks(self, chunks) -> list[Document]:
        return [
            Document(page_content=chunk, metadata={"belongs_to": self.file_name, "doc_number": i}) for i, chunk in
            enumerate(chunks)
        ]

    def add_metadata_to_summarized(self, summarized_chunks: list[str], ids_chunks: list[int]) -> list[Document]:
        return [Document(page_content=sum,
                         metadata={"doc_id": ids_chunks[i], "workspace_id": self.work_space_id,
                                   "belongs_to": self.file_name, "doc_number": i}) for i, sum
                in
                enumerate(summarized_chunks)]

    def get_documents_without_add_questions(self, documents: list[Document]) -> list[Document]:
        """Удаляет из сжатых текстов дополнительные вопросы, которые были добавлены перед векторизацией"""
        documents_without_questions = [
            Document(re.sub(r'Вопросы:.*?(?=\n\n|\Z)', '', summ.page_content, flags=re.DOTALL))
            for summ in documents]
        return documents_without_questions

    def _define_brief_max_word(self, context: str) -> int:
        len_context = len(context)
        if len_context <= 800:
            return 70
        elif 800 < len_context < 1600:
            return 100
        else:
            return 120

    def super_brief_content(self, documents: list[Document]) -> str:
        documents_content = [doc.page_content for doc in documents]
        context = "\n".join(documents_content)
        if len(context) <= 500:
            return context
        return self.model_service.get_super_brief_content(context, self._define_brief_max_word(context))

    def save_docs_and_add_in_retriever(self) -> (str, str):
        chunks = self.get_chunks()
        chunks_with_metadata = self.add_metadata_to_chunks(chunks)
        summarized_chunks = self.get_summarize_chunks(chunks)
        ids_chunks = DocumentsSaverService.save_chunks(self.user_id, self.work_space_id, chunks_with_metadata)
        summarized_chunks_with_metadata = self.add_metadata_to_summarized(summarized_chunks, ids_chunks)
        self.retriever.vectorstore.add_documents(summarized_chunks_with_metadata)
        return self.file_name, self.super_brief_content(
            self.get_documents_without_add_questions(summarized_chunks_with_metadata))

    @staticmethod
    def clear_vector_stores(user_id: int, workspace_id: int):
        """Удаляет векторное хранилище пользователя"""
        collection_name = f"user_{user_id}_{workspace_id}"
        client = chromadb.PersistentClient(path=rf"{VEC_BASES}\chroma_db_{user_id}")
        if collection_name in [name for name in client.list_collections()]:
            client.delete_collection(collection_name)

    @staticmethod
    def delete_file_from_vecstore(user_id: int, workspace_id: int, belongs_to: str):
        collection_name = f"user_{user_id}_{workspace_id}"
        client = chromadb.PersistentClient(path=rf"{VEC_BASES}\chroma_db_{user_id}")
        if collection_name in [name for name in client.list_collections()]:
            collection = client.get_collection(collection_name)
            collection.delete(where={"belongs_to": belongs_to})
