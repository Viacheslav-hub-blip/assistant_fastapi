from typing import List

from src.database.connection import session
from src.database.tables import Chunks
from sqlalchemy import and_


def insert_chunk(files: Chunks):
    with session() as s:
        s.add(files)
        s.commit()


def select_source_chunk(user_id: str, workspace_id: str, belongs_to: str, doc_number: str) -> Chunks:
    with session() as s:
        return s.query(Chunks).filter(
            and_(Chunks.user_id == user_id, Chunks.workspace_id == workspace_id, Chunks.belongs_to == belongs_to,
                 Chunks.doc_number == doc_number)).one()
