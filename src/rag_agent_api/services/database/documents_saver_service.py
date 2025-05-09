from typing import List
from src.database.tables import Chunks, Files
from src.database.connection import session
import datetime

from langchain.schema import Document


class DocumentsSaverService:
    @staticmethod
    def save_chunk(user_id: int, work_space_id: int, documents: List[Document]) -> list[int]:
        ids = []
        with session() as s:
            for doc in documents:
                chunk = Chunks(
                    user_id,
                    work_space_id,
                    doc.metadata["source_doc_name"],
                    doc.metadata["doc_number"],
                    doc.page_content
                )
                s.add(chunk)
                s.commit()
                s.flush()
                id = chunk.id
                ids.append(id)
        return ids

    @staticmethod
    def save_file(user_id: int, work_space_id: int, file_name: str, summary_content: str):
        with session() as s:
            file = Files(
                user_id,
                work_space_id,
                file_name,
                str(datetime.datetime.now()),
                summary_content
            )
            s.add(file)
            s.commit()
