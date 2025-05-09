from src.database.repositories import chunksCRUDRepository, filesCRUDRepository
from src.database.tables import Chunks


class DocumentsGetterService:
    @staticmethod
    def get_source_chunk(user_id: str, workspace_id: str, belongs_to: str, doc_number: str) -> Chunks:
        return chunksCRUDRepository.select_source_chunk(user_id, workspace_id, belongs_to, doc_number)

    @staticmethod
    def get_files_ids_names(user_id: str, workspace_id: int) -> dict[str, str]:
        files = filesCRUDRepository.select_all_by_user_id_and_work_space_id(user_id, workspace_id)
        result: dict[str, str] = {}
        for f in files:
            result[str(f.id)] = f.file_name
        return result

    @staticmethod
    def get_files_with_summary(user_id: str, workspace_id: int) -> dict[str, str]:
        files = filesCRUDRepository.select_all_by_user_id_and_work_space_id(user_id, workspace_id)
        result: dict[str, str] = {}
        for f in files:
            result[str(f.id)] = f.summary_content
        return result
