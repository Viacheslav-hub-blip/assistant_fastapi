from typing import List

from src.database.connection import session
from src.database.tables import WorkSpace


def insert_work_space(work_space: WorkSpace):
    with session() as s:
        s.add(work_space)
        s.commit()


def select_all_by_user_id(user_id: int) -> List[WorkSpace]:
    with session() as s:
        return s.query(WorkSpace).filter(WorkSpace.user_id == user_id).all()
