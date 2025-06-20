from typing import NamedTuple, List

from src.database.repositories import workSpaceCRUDRepository


class WorkSpace(NamedTuple):
    workspace_id: int
    user_id: int
    workspace_name: str


class WorkspacesService:
    @staticmethod
    def get_all_user_workspaces(user_id: int) -> List[WorkSpace]:
        return [WorkSpace(
            space.id,
            space.user_id,
            space.name)
            for space in workSpaceCRUDRepository.select_all_by_user_id(user_id)]

    @staticmethod
    def create_workspace(user_id: int, workspace_name: str) -> int:
        return workSpaceCRUDRepository.create_workspace(user_id, workspace_name)

    @staticmethod
    def check_exist_workspace(user_id: int, workspace_name: str) -> bool:
        """Проверяет наличие рабочего прсотранства пользователя с таким именем"""
        if workSpaceCRUDRepository.select_workspace(user_id, workspace_name):
            return True
        return False

    @staticmethod
    def delete_workspace(user_id: int, workspace_id: int):
        workSpaceCRUDRepository.delete_workspace(user_id, workspace_id)
