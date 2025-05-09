from typing import List
from src.database.tables import Chunks, Files
import datetime
from src.database.repositories import chunksCRUDRepository, filesCRUDRepository

from langchain.schema import Document


class DocumentsSaverService:
    @staticmethod
    def save_chunk(user_id: int, work_space_id: int, documents: List[Document]) -> list[int]:
        ids = []
        for doc in documents:
            chunk = Chunks(
                user_id=user_id,
                workspace_id=work_space_id,
                source_doc_name=doc.metadata["source_doc_name"],
                doc_number=doc.metadata["doc_number"],
                summary_content=doc.page_content
            )
            id = chunksCRUDRepository.insert_chunk(chunk)
            ids.append(id)
        return ids

    @staticmethod
    def save_file(user_id: int, work_space_id: int, file_name: str, summary_content: str):
        file = Files(
            user_id=user_id,
            workspace_id=work_space_id,
            file_name=file_name,
            load_date=str(datetime.datetime.now()),
            summary_content=summary_content
        )
        filesCRUDRepository.insert_file(file)
