from sqlalchemy import and_

from src.database.connection import session
from src.database.tables import Messages


def insert_messages(messages: Messages):
    with session as s:
        s.add(messages)
        print('добавлено сообщдение')
        s.commit()


def select_all_by_user_id_and_work_space_id(user_id: int, work_space_id: int) -> list[Messages]:
    with session as s:
        return s.query(Messages).filter(and_(Messages.user_id == user_id, Messages.workspace_id == work_space_id)).all()


def delete_all_messages_from_workspace(user_id: int, work_space_id: int):
    with session as s:
        s.query(Messages).filter(and_(Messages.user_id == user_id, Messages.workspace_id == work_space_id)).delete()
        s.commit()
