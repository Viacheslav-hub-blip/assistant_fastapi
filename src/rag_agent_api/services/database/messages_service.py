from typing import Literal, NamedTuple

from src.database.repositories.messagesCRUDRepository import insert_messages, select_all_by_user_id_and_work_space_id, \
    delete_all_messages_from_workspace
from src.database.tables import Messages

roles = Literal["user", "assistant", "jarvis"]


class Message(NamedTuple):
    type: str
    message: str
    source: str


class MessagesService:

    @staticmethod
    def insert_message(user_id: int, workspace_id: int, message: str, type: roles) -> None:
        new_message = Messages(
            user_id=user_id,
            workspace_id=workspace_id,
            message=message,
            message_type=type
        )
        insert_messages(new_message)

    @staticmethod
    def get_user_messages(user_id: int, workspace_id: int) -> list[Message]:
        messages = select_all_by_user_id_and_work_space_id(user_id, workspace_id)
        return [Message(mes.message_type, mes.message, "history") for mes in messages]

    @staticmethod
    def delete_messages(user_id: int, workspace_id: int) -> None:
        delete_all_messages_from_workspace(user_id, workspace_id)
