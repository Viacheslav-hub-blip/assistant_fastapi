from typing import NamedTuple, List, Any

# FastApi
from fastapi import APIRouter, UploadFile, File, Form

from src.rag_agent_api.agents.rag_agent import RagAgent
from src.rag_agent_api.config import TEMP_DOWNLOADS
# AGENTS
from src.rag_agent_api.langchain_model_init import model_for_answer
from src.rag_agent_api.langchain_model_init import model_for_brief_content
from src.rag_agent_api.services.database.documents_getter_service import DocumentsGetterService
from src.rag_agent_api.services.database.documents_remove_service import DocumentsRemoveService
from src.rag_agent_api.services.database.documents_saver_service import DocumentsSaverService
from src.rag_agent_api.services.database.messages_service import MessagesService, Message
from src.rag_agent_api.services.database.workspace_market_service import WorkspaceMarketService
from src.rag_agent_api.services.database.workspaces_service import WorkspacesService, WorkSpace
from src.rag_agent_api.services.llm_model_service import LLMModelService
from src.rag_agent_api.services.pdf_reader_service import PDFReader
# SERVICES
from src.rag_agent_api.services.retriever_service import VectorDBManager
from src.rag_agent_api.services.vectore_store_service import VecStoreService

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


async def format_agent_answer(answer) -> AgentAnswer:
    used_docs_names, used_docs = [], []
    print("answer in format", answer)
    question, generation = answer["question"], answer["answer"]
    if answer.get("used_docs", None):
        used_docs_names = answer["used_docs"]
        used_docs = [doc.page_content for doc in answer["neighboring_docs"]]
    return AgentAnswer(generation, used_docs_names, used_docs)


async def _invoke_agent(question: str, user_id: int, workspace_id: int, belongs_to: str,
                        chat_history: list[Message]) -> AgentAnswer:
    print("in _invoke_agent", user_id, workspace_id, belongs_to)
    retriever = VectorDBManager.get_or_create_retriever(user_id, workspace_id)
    rag_agent = RagAgent(model=model_for_answer, retriever=retriever)
    chat_history = [(mess.type, mess.message) for mess in chat_history]
    print("----------CHAT HISTORY----------", chat_history)

    result = rag_agent().invoke(
        {"question": question, "user_id": user_id, "workspace_id": workspace_id, "belongs_to": belongs_to,
         "chat_history": chat_history})
    return await format_agent_answer(result)


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


async def _get_doc_content(file_path: str) -> str | None:
    file_reader = PDFReader(file_path)
    content = file_reader.get_cleaned_content()

    print("content length:", len(content))
    if len(content) > 15000:
        return None
    return content


async def _save_doc_content(content: str, user_id: int,
                            file_name: str, work_space_id: int) -> (str, str):
    """Сохраняет извлеченную информацию"""
    retriever = VectorDBManager.get_or_create_retriever(user_id, work_space_id)
    vecstore_store_service = VecStoreService(llm_model_service, retriever, content, file_name, user_id, work_space_id)
    doc_id, summarize_content = vecstore_store_service.save_docs_and_add_in_retriever()
    return doc_id, summarize_content


@router.get("/")
async def get_answer(question: str, user_id: int, workspace_id: int, belongs_to: str = None) -> AgentAnswer:
    print(question, user_id, workspace_id)
    MessagesService.insert_message(user_id, workspace_id, question, "user")
    chat_history = MessagesService.get_user_messages(user_id, workspace_id)
    belongs_to = belongs_to if belongs_to != 'null' else None
    answer = await _invoke_agent(question, user_id, workspace_id, belongs_to, chat_history)
    MessagesService.insert_message(user_id, workspace_id, answer.answer, "assistant")
    return answer


@router.post("/load_file")
async def load_file(file: UploadFile = File(...), user_id: int = Form(...), workspace_id: int = Form(...)) -> dict[
    str, Any]:
    destination = await _save_file_local(user_id, workspace_id, file)
    content = await _get_doc_content(destination)
    if content:
        doc_id, summarize_content = await _save_doc_content(content, user_id, file.filename, workspace_id)
        DocumentsSaverService.save_file(user_id, workspace_id, file.filename, summarize_content)
        return {"satus": 200, "doc_id": doc_id, "summary": summarize_content}
    return {"satus": 400, "error": "слишком большой файл"}


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


@router.get("/delete_workspace")
async def delete_workspace(user_id: int, workspace_id: int) -> str:
    await delete_all_files(user_id, workspace_id)
    MessagesService.delete_messages(user_id, workspace_id)
    WorkspacesService.delete_work_space(user_id, workspace_id)
    return "Рабочее пространство удалено"


@router.get("/clear_chat_history")
async def clear_chat_history(user_id: int, workspace_id: int) -> dict[str, int]:
    MessagesService.delete_messages(user_id, workspace_id)
    return {"status": 200}


@router.get("/delete_file")
async def delete_file(user_id: int, workspace_id: int, file_id: int, file_name: str) -> dict[str, Any]:
    DocumentsRemoveService.delete_document_by_id(user_id, workspace_id, file_id)
    VecStoreService.delete_file_from_vecstore(user_id, workspace_id, file_name)
    return {"status": 200}


@router.get('/user_workspaces')
async def user_workspaces(user_id: int) -> list[WorkSpace]:
    return WorkspacesService.get_all_user_workspaces(user_id)


@router.get("/create_new_workspace")
async def create_new_workspace(user_id: int, workspace_name: str) -> int:
    return WorkspacesService.create_workspace(user_id, workspace_name)


@router.get("/get_messages")
async def get_user_messages(user_id: int, workspace_id: int) -> list[dict[str, Any]]:
    messages = MessagesService.get_user_messages(user_id, workspace_id)
    print(messages)
    return [message._asdict() for message in messages]


@router.post("/copy_workspace")
async def copy_workspace(source_user_id: int, source_workspace_id: int, target_user_id: int,
                         target_workspace_name: str) -> dict[str, Any]:
    if not WorkspacesService.check_exist_workspace(target_user_id, target_workspace_name):
        target_workspace_id = WorkspacesService.create_workspace(target_user_id, target_workspace_name)
        VectorDBManager.copy_collection(source_user_id, source_workspace_id, target_user_id, target_workspace_id)
        all_chunks = DocumentsGetterService.get_all_chunks_from_workspace(source_user_id, source_workspace_id)
        all_files = DocumentsGetterService.get_all_files_from_workspace(source_user_id, source_workspace_id)

        DocumentsSaverService.save_chunks(target_user_id, target_workspace_id, all_chunks)
        DocumentsSaverService.save_many_files(target_user_id, target_workspace_id, all_files)
        return {"status": "sucsess", "user_id": target_user_id, "workspace_id": target_workspace_id}
    return {"status": "fail"}


@router.get("/save_message")
async def save_message(user_id: int, workspace_id: int, message: str, type: str) -> None:
    MessagesService.insert_message(user_id, workspace_id, message, type)


@router.post("/load_workspace_to_market")
async def load_workspace_to_market(
        user_id: int,
        source_workspace_id: int,
        workspace_name: str,
        workspace_description: str
) -> dict[str, int]:
    if not WorkspaceMarketService.select_workspace_by_user_id_and_name(user_id, workspace_name):
        space_id = WorkspaceMarketService.insert_workspace_in_market(
            user_id, source_workspace_id, workspace_name, workspace_description
        )

        if isinstance(space_id, int):
            return {"status": 200}
    return {"status": 404}


@router.get("/workspaces_in_market")
async def get_workspaces_in_market():
    dict = [workspace._asdict() for workspace in WorkspaceMarketService.select_all_workspaces_inmarket()]
    return dict
