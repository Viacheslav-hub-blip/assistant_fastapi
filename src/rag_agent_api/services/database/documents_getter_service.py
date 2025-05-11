from src.database.repositories import chunksCRUDRepository, filesCRUDRepository
from src.database.tables import Chunks
from langchain_core.documents import Document


class DocumentsGetterService:
    @staticmethod
    def get_source_chunk(user_id: int, workspace_id: int, belongs_to: str, doc_number: str) -> Document:
        
        """Извлечение исходного фрагмента текста
        belongs_to  - название документа, к которому принадлежит краткое содержание 
        doc_number  - номер фрагмента

        returns: Document, который содержит текст исходного фрагмента и metadata с его принадлежностью и позицией
        """
        chunk = chunksCRUDRepository.select_source_chunk(user_id, workspace_id, belongs_to, doc_number)
        if chunk:
            print("source chunk from db", chunk)
            return Document(page_content=chunk.summary_content,
                            metadata={"belongs_to": chunk.source_doc_name, "doc_number": chunk.doc_number})
        else:
            return Document(page_content="")

    @staticmethod
    def get_files_ids_names(user_id: int, workspace_id: int) -> dict[str, str]:
        """
        Возвращает все id и названия файлов пользователя в workspace
        {"id": "name"}
        """
        files = filesCRUDRepository.select_all_by_user_id_and_work_space_id(user_id, workspace_id)
        result: dict[str, str] = {}
        for f in files:
            result[str(f.id)] = f.file_name
        return result

    @staticmethod
    def get_files_with_summary(user_id: int, workspace_id: int) -> dict[str, str]:
        """Возвращает загруженные файлы с их кратких содержанием """
        {"id": "summary"}
        files = filesCRUDRepository.select_all_by_user_id_and_work_space_id(user_id, workspace_id)
        result: dict[str, str] = {}
        for f in files:
            result[str(f.id)] = f.summary_content
        return result
