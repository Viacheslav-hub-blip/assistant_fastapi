from typing import List

from src.database.connection import session
from src.database.tables import FavoriteMessages


def add_in_favorite(id: int, user_id: int, workspace_id: int, text: str) -> int:
    message = FavoriteMessages(id=id, user_id=user_id, workspace_id=workspace_id, text=text)
    with session as s:
        s.add(message)
        s.commit()
        s.flush()
        return message.id


def delete_from_favorite(id: int, user_id: int, workspace_id: int):
    with session() as s:
        return s.query(FavoriteMessages).filter(
            FavoriteMessages.user_id == user_id,
            FavoriteMessages.workspace_id == workspace_id,
            FavoriteMessages.id == id
        ).delete()


def select_all_favorite_messages_from_workspace(user_id: int, workspace_id: int) -> List[FavoriteMessages]:
    with session as s:
        return s.query(FavoriteMessages).filter(FavoriteMessages.user_id == user_id,
                                                FavoriteMessages.workspace_id == workspace_id).all()
