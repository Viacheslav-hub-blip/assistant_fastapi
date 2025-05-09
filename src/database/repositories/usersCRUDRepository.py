from typing import List

from src.database.connection import session
from src.database.tables import Users


def insert_user(user: Users) -> int:
    with session() as s:
        s.add(user)
        s.commit()
        s.flush()
        return user.id


def select_all() -> List[Users]:
    with session() as s:
        return s.query(Users).all()
