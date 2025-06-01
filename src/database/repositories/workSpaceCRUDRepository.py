from typing import List

from src.database.connection import session
from src.database.tables import WorkSpace


def create_workspace(user_id: int, workspace_name: str) -> int:
    space = WorkSpace(user_id=user_id, name=workspace_name)
    with session as s:
        s.add(space)
        s.commit()
        print("mew workspace id", space.id)
        return space.id


def select_all_by_user_id(user_id: int) -> List[WorkSpace]:
    with session as s:
        return s.query(WorkSpace).filter(WorkSpace.user_id == user_id).all()


def select_workspace(user_id: int, workspace_name: str) -> WorkSpace | None:
    with session as s:
        return s.query(WorkSpace).filter(WorkSpace.user_id == user_id, WorkSpace.name == workspace_name).first()
