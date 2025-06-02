from typing import NamedTuple

from src.database.repositories import usersCRUDRepository


class User(NamedTuple):
    id: int
    email: str
    login: str
    password: str


class UserService:
    @staticmethod
    def select_user_by_email(user_email: str) -> User | None:
        return usersCRUDRepository.select_user_by_email(user_email)

    @staticmethod
    def insert_user(email: str, login: str, password: str) -> int:
        return usersCRUDRepository.insert_user(email, login, password)

    @staticmethod
    def check_user_exists(email: str) -> bool:
        return UserService.select_user_by_email(email) is not None
