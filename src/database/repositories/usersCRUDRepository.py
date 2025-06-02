from typing import List

from src.database.connection import session
from src.database.tables import Users


def insert_user(email: str, login: str, password: str) -> int:
    user = Users(email=email, login=login, password=password)
    with session as s:
        s.add(user)
        s.commit()
        s.flush()
        return user.id


def select_user_by_email(user_email: str) -> Users | None:
    with session as s:
        return s.query(Users).filter(Users.email == user_email).first()


def select_all() -> List[Users]:
    with session as s:
        return s.query(Users).all()
