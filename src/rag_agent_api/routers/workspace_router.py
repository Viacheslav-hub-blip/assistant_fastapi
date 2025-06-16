from typing import Any

from fastapi import APIRouter

from src.rag_agent_api.services.database.documents_getter_service import DocumentsGetterService
from src.rag_agent_api.services.database.documents_remove_service import DocumentsRemoveService
from src.rag_agent_api.services.database.documents_saver_service import DocumentsSaverService
from src.rag_agent_api.services.database.messages_service import MessagesService
from src.rag_agent_api.services.database.workspace_market_service import WorkspaceMarketService
from src.rag_agent_api.services.database.workspaces_service import WorkspacesService, WorkSpace
from src.rag_agent_api.services.retriever_service import VectorDBManager
from src.rag_agent_api.services.vectore_store_service import VecStoreService

router = APIRouter(
    prefix="/workspace",
    tags=["agent"],
)


@router.get("/delete_workspace")
async def delete_workspace(user_id: int, workspace_id: int) -> str:
    VecStoreService.clear_vector_stores(user_id, workspace_id)
    DocumentsRemoveService.delete_all_files_in_workspace(user_id, workspace_id)
    DocumentsRemoveService.delete_all_chunks_in_workspace(user_id, workspace_id)
    MessagesService.delete_messages(user_id, workspace_id)
    WorkspacesService.delete_workspace(user_id, workspace_id)
    return "Рабочее пространство удалено"


@router.get('/user_workspaces')
async def user_workspaces(user_id: int) -> list[WorkSpace]:
    return WorkspacesService.get_all_user_workspaces(user_id)


@router.get("/create_new_workspace")
async def create_new_workspace(user_id: int, workspace_name: str) -> int:
    return WorkspacesService.create_workspace(user_id, workspace_name)


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
    dict = [workspace._asdict() for workspace in WorkspaceMarketService.select_all_workspaces_in_market()]
    return dict
