from typing import List

from src.database.connection import session
from src.database.tables import Chunks
from sqlalchemy import and_


def insert_chunk(chunk: Chunks) -> int:
    with session as s:
        s.add(chunk)
        s.commit()
        s.flush()
        return chunk.id


def select_source_chunk(user_id: int, workspace_id: int, belongs_to: str, doc_number: str) -> Chunks | None:
    with session as s:
        res = s.query(Chunks).filter(
            and_(Chunks.user_id == user_id, Chunks.workspace_id == workspace_id, Chunks.source_doc_name == belongs_to,
                 Chunks.doc_number == doc_number)).all()
        if len(res) != 0:
            return res[0]
        return None


def select_all_chunks_from_workspace(user_id: int, workspace_id: int) -> list[Chunks]:
    with session as s:
        res = s.query(Chunks).filter(
            and_(Chunks.user_id == user_id, Chunks.workspace_id == workspace_id)
        ).all()
    return res
