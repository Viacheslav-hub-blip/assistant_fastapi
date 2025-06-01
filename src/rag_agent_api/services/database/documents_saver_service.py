import datetime
from typing import List, Optional, NamedTuple

from langchain.schema import Document

from src.database.repositories import chunksCRUDRepository, filesCRUDRepository
from src.database.tables import Chunks, Files


class File(NamedTuple):
    user_id: int
    worksapce_id: int
    file_name: str
    load_date: str
    summary_content: str


class DocumentsSaverService:
    @staticmethod
    def save_chunks(user_id: int, work_space_id: int, documents: List[Document]) -> list[int]:
        ids = []
        for doc in documents:
            chunk = Chunks(
                user_id=user_id,
                workspace_id=work_space_id,
                source_doc_name=doc.metadata["belongs_to"],
                doc_number=doc.metadata["doc_number"],
                summary_content=doc.page_content
            )
            id = chunksCRUDRepository.insert_chunk(chunk)
            ids.append(id)
        return ids

    @staticmethod
    def save_file(user_id: int, work_space_id: int, file_name: str, summary_content: str,
                  load_date: Optional[str] = None) -> None:
        file = Files(
            user_id=user_id,
            workspace_id=work_space_id,
            file_name=file_name,
            load_date=load_date if load_date else str(datetime.datetime.now()),
            summary_content=summary_content
        )
        filesCRUDRepository.insert_file(file)

    @staticmethod
    def save_many_files(user_id: int, workspace_id: int, files: list[File]) -> None:
        for file in files:
            DocumentsSaverService.save_file(
                user_id,
                workspace_id,
                file.file_name,
                file.summary_content,
                file.load_date
            )
