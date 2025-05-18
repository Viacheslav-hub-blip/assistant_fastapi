from typing import NamedTuple, List

from src.rag_agent_api.config import TEMP_DOWNLOADS
# FastApi
from fastapi import APIRouter, UploadFile, File, Form
# SERVICES
from src.rag_agent_api.services.retriever_service import RetrieverSrvice
from src.rag_agent_api.services.database.documents_saver_service import DocumentsSaverService
from src.rag_agent_api.services.database.documents_getter_service import DocumentsGetterService
from src.rag_agent_api.services.database.documents_remove_service import DocumentsRemoveService
from src.rag_agent_api.services.database.workspaces_service import WorkspacesService, WorkSpace
from src.rag_agent_api.services.database.messages_service import MessagesService
from src.rag_agent_api.services.pdf_reader_service import PDFReader
from src.rag_agent_api.services.vectore_store_service import VecStoreService
from src.rag_agent_api.services.llm_model_service import LLMModelService
# AGENTS
from src.rag_agent_api.langchain_model_init import model_for_answer
from src.rag_agent_api.agents.rag_agent import RagAgent
from src.rag_agent_api.langchain_model_init import model_for_brief_content
from langchain_core.documents import Document

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)
llm_model_service = LLMModelService(model_for_brief_content)


class DocWithIdAndSummary(NamedTuple):
    id: str
    name: str
    summary: str


class AgentAnswer(NamedTuple):
    answer: str
    used_docs_names: List[str]
    used_docs: List[str]


class Message(NamedTuple):
    type: str
    message: str


async def _invoke_agent(question: str, user_id: int, workspace_id: int, belongs_to: str) -> AgentAnswer:
    used_docs_names, used_docs = [], []
    retriever = RetrieverSrvice.get_or_create_retriever(user_id, workspace_id)
    rag_agent = RagAgent(model=model_for_answer, retriever=retriever)
    result = rag_agent().invoke(
        {"question": question, "user_id": user_id, "workspace_id": workspace_id, "belongs_to": belongs_to})
    question, generation = result["question"], result["answer"]
    if result["used_docs"]:
        used_docs_names = result["used_docs"]
        used_docs = [doc.page_content for doc in result["neighboring_docs"]]
    return AgentAnswer(generation, used_docs_names, used_docs)


async def _save_file_local(user_id: int, work_space_id: int, file) -> str:
    try:
        contents = file.file.read()
        file_name = f"{user_id}_{work_space_id}_{file.filename}"
        type = 'pdf'
        destination = rf"{TEMP_DOWNLOADS}\{file_name}.{type}"
        with open(destination, 'wb') as f:
            f.write(contents)
        return destination
    finally:
        file.file.close()


async def _get_doc_content(file_path: str):
    file_reader = PDFReader(file_path)
    content = file_reader.get_cleaned_content()
    print("content length:", len(content))
    return content


async def _save_doc_content(content: str, user_id: int,
                            file_name: str, work_space_id: int) -> (str, str):
    """Сохраняет извлеченную информацию"""
    retriever = RetrieverSrvice.get_or_create_retriever(user_id, work_space_id)
    vecstore_store_service = VecStoreService(llm_model_service, retriever, content, file_name, user_id, work_space_id)
    doc_id, summarize_content = vecstore_store_service.save_docs_and_add_in_retriever()
    return doc_id, summarize_content


@router.get("/")
async def get_answer(question: str, user_id: int, workspace_id: int, belongs_to: str = None) -> AgentAnswer:
    print("id", belongs_to)
    answer = await _invoke_agent(question, user_id, workspace_id, belongs_to)
    print("answer", answer)
    return answer


@router.post("/load_file")
async def load_file(file: UploadFile = File(...), user_id: int = Form(...), work_space_id: int = Form(...)):
    destination = await _save_file_local(user_id, work_space_id, file)
    content = await _get_doc_content(destination)
    doc_id, summarize_content = await _save_doc_content(content, user_id, file.filename, work_space_id)
    DocumentsSaverService.save_file(user_id, work_space_id, file.filename, summarize_content)
    return {"doc_id": doc_id, "summary": summarize_content}


@router.get("/my_files")
async def my_files(user_id: int, workspace_id: int) -> list[DocWithIdAndSummary]:
    all_files_ids_names = DocumentsGetterService.get_files_ids_names(user_id, workspace_id)
    files_summary = DocumentsGetterService.get_files_with_summary(user_id, workspace_id)
    docs = [DocWithIdAndSummary(k, v, files_summary[k]) for k, v in all_files_ids_names.items()]
    return docs


@router.get("/delete_all_files")
async def delete_all_files(user_id: int, workspace_id: int):
    VecStoreService.clear_vector_stores(user_id, workspace_id)
    DocumentsRemoveService.delete_all_files_in_workspace(user_id, workspace_id)
    return "Загруженные документы удалены"


@router.get("/delete_file")
async def delete_file(user_id: int, workspace_id: int, file_id: int, file_name: str):
    DocumentsRemoveService.delete_document_by_id(user_id, workspace_id, file_id)
    VecStoreService.delete_file_from_vecstore(user_id, workspace_id, file_name)
    return "Файл удален"


@router.get('/user_workspaces')
async def user_workspaces(user_id: int) -> list[WorkSpace]:
    return WorkspacesService.get_all_user_workspaces(user_id)


@router.get("/create_new_workspace")
async def create_new_workspace(user_id: int, workspace_name: str) -> int:
    return WorkspacesService.create_workspace(user_id, workspace_name)


@router.get("/save_message")
async def save_message(user_id: int, workspace_id: int, message: str, type: str) -> None:
    MessagesService.insert_message(user_id, workspace_id, message, type)


@router.get("/get_messages")
async def get_user_messages(user_id: int, workspace_id: int) -> list[Message]:
    messages = MessagesService.get_user_messages(user_id, workspace_id)
    return [Message(mes.message, mes.message_type) for mes in messages]
