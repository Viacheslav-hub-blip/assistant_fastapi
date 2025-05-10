from typing import List

from src.database.connection import session
from src.database.tables import Files
from sqlalchemy import and_


def select_all_by_user_id_and_work_space_id(user_id: int, work_space_id: int) -> List[Files]:
    with session() as s:
        return s.query(Files).filter(and_(Files.user_id == user_id, Files.workspace_id == work_space_id)).all()


def insert_file(file: Files):
    with session as s:
        s.add(file)
        s.commit()
