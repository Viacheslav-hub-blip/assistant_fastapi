from typing import NamedTuple, Any

from fastapi import APIRouter, UploadFile, File, Form

from src.rag_agent_api.config import TEMP_DOWNLOADS
from src.rag_agent_api.langchain_model_init import model_for_brief_content
from src.rag_agent_api.services.database.documents_getter_service import DocumentsGetterService
from src.rag_agent_api.services.database.documents_remove_service import DocumentsRemoveService
from src.rag_agent_api.services.database.documents_saver_service import DocumentsSaverService
from src.rag_agent_api.services.llm_model_service import LLMModelService
from src.rag_agent_api.services.pdf_reader_service import PDFReader
from src.rag_agent_api.services.retriever_service import VectorDBManager
from src.rag_agent_api.services.vectore_store_service import VecStoreService

router = APIRouter(
    prefix="/files",
    tags=["files"],
)

llm_model_service = LLMModelService(model_for_brief_content)


class DocWithIdAndSummary(NamedTuple):
    id: str
    name: str
    summary: str


async def _save_file_local(user_id: int, work_space_id: int, file) -> str:
    try:
        contents = file.file.read()
        file_name = f"{user_id}_{work_space_id}_{file.filename}"
        destination = rf"{TEMP_DOWNLOADS}\{file_name}"
        with open(destination, 'wb') as f:
            f.write(contents)
        return destination
    finally:
        file.file.close()


async def _get_doc_content(file_path: str) -> str | None:
    file_reader = PDFReader(file_path)
    content = file_reader.get_cleaned_content()
    if len(content) > 15000:
        return None
    return content


async def _save_doc_content(content: str, user_id: int,
                            file_name: str, work_space_id: int) -> tuple[str, str] | Exception:
    """Сохраняет извлеченную информацию"""
    retriever = VectorDBManager.get_or_create_retriever(user_id, work_space_id)
    vecstore_store_service = VecStoreService(llm_model_service, retriever, content, file_name, user_id, work_space_id)

    try:
        doc_id, summarize_content = vecstore_store_service.save_docs_and_add_in_retriever()
        return doc_id, summarize_content
    except Exception as e:
        return e


@router.post("/load_file")
async def load_file(
        file: UploadFile = File(...),
        user_id: int = Form(...),
        workspace_id: int = Form(...)) -> dict[str, Any]:
    destination = await _save_file_local(user_id, workspace_id, file)
    content = await _get_doc_content(destination)
    if content:
        try:
            doc_id, summarize_content = await _save_doc_content(content, user_id, file.filename, workspace_id)
            DocumentsSaverService.save_file(user_id, workspace_id, file.filename, summarize_content)
            return {"status": 200, "doc_id": doc_id, "summary": summarize_content}
        except Exception as e:
            return {"status": 400, "error": str(e)}
    return {"status": 400, "error": "слишком большой файл"}


@router.get("/my_files")
async def my_files(user_id: int, workspace_id: int) -> list[DocWithIdAndSummary]:
    all_files_ids_names = DocumentsGetterService.get_files_ids_names(user_id, workspace_id)
    files_summary = DocumentsGetterService.get_files_with_summary(user_id, workspace_id)
    docs = [DocWithIdAndSummary(k, v, files_summary[k]) for k, v in all_files_ids_names.items()]
    return docs


@router.get("/delete_all_files")
async def delete_all_files(user_id: int, workspace_id: int) -> str:
    VecStoreService.clear_vector_stores(user_id, workspace_id)
    DocumentsRemoveService.delete_all_files_in_workspace(user_id, workspace_id)
    DocumentsRemoveService.delete_all_chunks_in_workspace(user_id, workspace_id)
    return "Загруженные документы удалены"


@router.get("/delete_file")
async def delete_file(user_id: int, workspace_id: int, file_id: int, file_name: str) -> dict[str, Any]:
    DocumentsRemoveService.delete_file_by_id(user_id, workspace_id, file_id)
    VecStoreService.delete_file_from_vecstore(user_id, workspace_id, file_name)
    return {"status": 200}
