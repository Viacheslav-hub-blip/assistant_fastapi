from typing import NamedTuple

from src.database.repositories.workSpacesMarketCRUDRepository import insert_worksapce, select_all_worksapces, select_workspace_by_user_id_and_name


class WorkspaceMarket(NamedTuple):
    user_id: int
    source_workspace_id: int
    workspace_name: str
    workspace_description: str


class WorkspaceMarketService:
    @staticmethod
    def insert_workspace_in_market(
            user_id: int,
            source_workspace_id: int,
            workspace_name: str,
            workspace_description: str
    ) -> int:
        return insert_worksapce(user_id, source_workspace_id, workspace_name, workspace_description)

    @staticmethod
    def select_all_workspaces_inmarket() -> list[WorkspaceMarket]:
        spaces_in_market = select_all_worksapces()
        return [WorkspaceMarket(space.user_id, space.source_workspace_id, space.workspace_name, space.workspace_description)
                for space in spaces_in_market]

    @staticmethod
    def select_workspace_by_user_id_and_name(user_id: int, workspace_name: str) -> WorkspaceMarket | None:
        return select_workspace_by_user_id_and_name(user_id, workspace_name)
