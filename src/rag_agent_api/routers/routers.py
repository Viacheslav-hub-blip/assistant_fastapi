from typing import NamedTuple

from src.rag_agent_api.config import TEMP_DOWNLOADS
# FastApi
from fastapi import APIRouter, UploadFile, HTTPException, File, Form
# SERVICES
from src.rag_agent_api.services.retriever_service import RetrieverSrvice
from src.rag_agent_api.services.documents_saver_service import DocumentsSaver
from src.rag_agent_api.services.documents_getter_service import DocumentsGetterService
from src.rag_agent_api.services.pdf_reader_service import PDFReader
from src.rag_agent_api.services.vectore_store_service import VecStoreService
from src.rag_agent_api.services.llm_model_service import LLMModelService
# AGENTS
from src.rag_agent_api.langchain_model_init import model_for_answer
from src.rag_agent_api.agents.rag_agent import RagAgent
from src.rag_agent_api.langchain_model_init import model_for_brief_content

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)
llm_model_service = LLMModelService(model_for_brief_content)


class DocWithIdAndSummary(NamedTuple):
    id: str
    name: str
    summary: str


async def _invoke_agent(question: str, user_id: str) -> str:
    retriever = RetrieverSrvice.get_or_create_retriever(user_id)
    rag_agent = RagAgent(model=model_for_answer, retriever=retriever)
    result = rag_agent().invoke({"question": question, "user_id": user_id, "file_metadata_id": None})
    question, generation = result["question"], result["answer"]
    return generation


async def _save_file(file) -> str:
    try:
        contents = file.file.read()
        file_id = hash(file.filename)
        type = 'pdf'
        destination = rf"{TEMP_DOWNLOADS}\{file_id}.{type}"
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


async def _save_doc_content(content: str, user_id: str,
                            file_name: str) -> (str, str):
    """Сохраняет извлеченную информацию"""
    retriever = RetrieverSrvice.get_or_create_retriever(user_id)
    vecstore_store_service = VecStoreService(llm_model_service, retriever, content, file_name)
    doc_id, summarize_content = vecstore_store_service.save_docs_and_add_in_retriever()
    DocumentsSaver.save_doc_summary(user_id, doc_id, summarize_content)
    return doc_id, summarize_content


@router.get("/")
async def get_answer(question: str, user_id: str):
    answer = await _invoke_agent(question, user_id)
    print("answer", answer)
    return answer


@router.post("/load_file")
async def load_file(file: UploadFile = File(...), user_id: str = Form(...)):
    destination = await _save_file(file)
    content = await _get_doc_content(destination)
    doc_id, summarize_content = await _save_doc_content(content, user_id, file.filename)
    return {"doc_id": doc_id, "summary": summarize_content}


@router.get("/my_files")
async def my_files(user_id: str) -> list[DocWithIdAndSummary]:
    all_files_ids_names = DocumentsGetterService.get_files_ids_names(user_id)
    files_summary = DocumentsGetterService.get_files_summary(user_id)
    docs = [DocWithIdAndSummary(k, v, files_summary[k]) for k, v in all_files_ids_names.items()]
    return docs


@router.get("/delete_all_files")
async def delete_all_files(user_id: str):
    VecStoreService.clear_vector_stores(user_id)
    DocumentsSaver.clear_user_directory(user_id)
    return "Загруженные документы удалены"
