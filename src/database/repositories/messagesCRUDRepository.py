from typing import List

from src.database.connection import session
from src.database.tables import Messages
from sqlalchemy import and_


def insert_messages(messages: Messages):
    with session() as s:
        s.add(messages)
        s.commit()


def select_all_by_user_id_and_work_space_id(user_id: str, work_space_id: int) -> List[Messages]:
    with session() as s:
        return s.query(Messages).filter(and_(Messages.user_id == user_id, Messages.workspace_id == work_space_id)).all()
