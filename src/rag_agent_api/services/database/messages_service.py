from typing import Literal, NamedTuple

from src.database.repositories.favoriteAnswersCrudRepository import (
    add_in_favorite,
    delete_from_favorite,
    select_all_favorite_messages,

)
from src.database.repositories.messagesCRUDRepository import (
    insert_messages,
    select_all_by_user_id_and_work_space_id,
    delete_all_messages_from_workspace,
    update_favorite_status_in_history
)
from src.database.tables import Messages

roles = Literal["user", "assistant", "jarvis"]


class Message(NamedTuple):
    id: int
    type: str
    message: str
    source: str
    infavorite: bool


class FavoriteMessage(NamedTuple):
    id: int
    user_id: int
    workspace_id: int
    message: str


class MessagesService:

    @staticmethod
    def insert_message(user_id: int, workspace_id: int, message: str, type: roles) -> int:
        new_message = Messages(
            user_id=user_id,
            workspace_id=workspace_id,
            message=message,
            message_type=type,
            infavorite=False
        )
        print("new message", new_message)
        return insert_messages(new_message)

    @staticmethod
    def get_user_messages(user_id: int, workspace_id: int) -> list[Message]:
        messages = select_all_by_user_id_and_work_space_id(user_id, workspace_id)
        return [Message(mes.id, mes.message_type, mes.message, "history", mes.infavorite) for mes in messages]

    @staticmethod
    def delete_messages(user_id: int, workspace_id: int) -> None:
        delete_all_messages_from_workspace(user_id, workspace_id)

    @staticmethod
    def add_in_favorite(id: int, user_id: int, workspace_id: int, text: str) -> None:
        add_in_favorite(id, user_id, workspace_id, text)

    @staticmethod
    def delete_from_favorites(id: int, user_id: int, workspace_id: int) -> None:
        delete_from_favorite(id, user_id, workspace_id)

    @staticmethod
    def select_all_favorite_messages(
            user_id: int,
    ) -> list[FavoriteMessage]:
        messages = select_all_favorite_messages(user_id)
        return [FavoriteMessage(mes.id, mes.user_id, mes.workspace_id, mes.text) for mes in messages]

    @staticmethod
    def update_favorite_status_in_history(id: int, user_id: int, workspace_id: int, status: bool):
        update_favorite_status_in_history(id, user_id, workspace_id, status)
