from src.database.connection import session
from src.database.tables import WorkspacesMarket


def insert_worksapce(
        user_id: int,
        source_workspace_id: int,
        workspace_name: str,
        workspace_description: str
) -> int:
    space = WorkspacesMarket(
        user_id=user_id,
        source_workspace_id=source_workspace_id,
        workspace_name=workspace_name,
        workspace_description=workspace_description
    )

    with session as s:
        s.add(space)
        s.commit()
        print("новая позиця в маркете", space.id)
        return space.id


def select_all_worksapces() -> list[WorkspacesMarket]:
    with session as s:
        return s.query(WorkspacesMarket).all()


def select_workspace_by_user_id_and_name(user_id: int, workspace_name: str) -> WorkspacesMarket | None:
    with session as s:
        return s.query(WorkspacesMarket).filter(WorkspacesMarket.user_id == user_id,
                                                WorkspacesMarket.workspace_name == workspace_name).first()


def delete_workspace_from_market(user_id: int, workspace_id: int):
    print("удаление пространство из маркета")
    with session as s:
        s.query(WorkspacesMarket).filter(WorkspacesMarket.user_id == user_id,
                                         WorkspacesMarket.source_workspace_id == workspace_id).delete()
        s.commit()
