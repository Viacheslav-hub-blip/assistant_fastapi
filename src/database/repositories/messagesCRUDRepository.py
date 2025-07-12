from sqlalchemy import and_

from src.database.connection import session
from src.database.tables import Messages


def insert_messages(messages: Messages) -> int:
    m = messages
    with session as s:
        s.add(m)
        s.commit()
        return m.id


def select_all_by_user_id_and_work_space_id(user_id: int, work_space_id: int) -> list[Messages]:
    with session as s:
        return (s.query(Messages)
                .filter(
            and_(Messages.user_id == user_id, Messages.workspace_id == work_space_id))
                .order_by(Messages.id)
                .all())


def delete_all_messages_from_workspace(user_id: int, work_space_id: int):
    with session as s:
        s.query(Messages).filter(and_(Messages.user_id == user_id, Messages.workspace_id == work_space_id)).delete()
        s.commit()


def update_favorite_status_in_history(id: int, user_id: int, workspace_id: int, status: bool):
    with session as s:
        s.query(Messages).filter(
            Messages.id == id,
            Messages.user_id == user_id,
            Messages.workspace_id == workspace_id).update({"infavorite": status})
        s.commit()
