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
