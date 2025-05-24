from src.database.repositories.messagesCRUDRepository import insert_messages, select_all_by_user_id_and_work_space_id
from src.database.tables import Messages
from typing import Literal, NamedTuple

roles = Literal["user", "assistant", "jarvis"]


class Message(NamedTuple):
    type: str
    message: str


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
        return [Message(mes.message_type, mes.message) for mes in messages]
