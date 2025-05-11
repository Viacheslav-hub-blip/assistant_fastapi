from src.database.repositories import filesCRUDRepository


class DocumentsRemoveService:
    @staticmethod
    def delete_document_by_id(user_id: int, workspace_id: int, file_id: int) -> None:
        return filesCRUDRepository.delete_file_by_id(user_id, workspace_id, file_id)

    @staticmethod
    def delete_all_files_in_workspace(user_id: int, workspace_id: int) -> None:
        return filesCRUDRepository.delete_all_files_in_workspace(user_id, workspace_id)
