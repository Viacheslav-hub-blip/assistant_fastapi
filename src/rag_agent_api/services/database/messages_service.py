from src.database.repositories.messagesCRUDRepository import insert_messages, select_all_by_user_id_and_work_space_id
from src.database.tables import Messages


class MessagesService:

    @staticmethod
    def insert_message(user_id: int, workspace_id: int, message: str, type: str) -> None:
        new_message = Messages(
            user_id=user_id,
            workspace_id=workspace_id,
            message=message,
            message_type=type
        )
        insert_messages(new_message)

    @staticmethod
    def get_user_messages(user_id: int, workspace_id: int) -> list[Messages]:
        return select_all_by_user_id_and_work_space_id(user_id, workspace_id)
