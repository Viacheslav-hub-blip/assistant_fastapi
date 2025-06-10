from src.database.repositories import filesCRUDRepository, chunksCRUDRepository


class DocumentsRemoveService:
    @staticmethod
    def delete_file_by_id(user_id: int, workspace_id: int, file_id: int) -> None:
        """Удаляет документв пространстве по id документа"""
        return filesCRUDRepository.delete_file_by_id(user_id, workspace_id, file_id)

    @staticmethod
    def delete_all_files_in_workspace(user_id: int, workspace_id: int) -> None:
        """Удаляет все файлы в пространстве"""
        return filesCRUDRepository.delete_all_files_in_workspace(user_id, workspace_id)

    @staticmethod
    def delete_all_chunks_in_workspace(user_id: int, workspace_id: int) -> None:
        """Удаляет все фрагменты в пространстве"""
        return chunksCRUDRepository.delete_all_chunks_in_workspace(user_id, workspace_id)
