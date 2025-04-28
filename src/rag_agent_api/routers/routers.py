# FastApi
from fastapi import APIRouter, UploadFile, HTTPException
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
        destination = rf"/home/alex/PycharmProjects/pythonProject/src/temp_downloads/{file_id}.{type}"
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
                            file_name: str) -> str:
    """Сохраняет извлеченную информацию"""
    retriever = RetrieverSrvice.get_or_create_retriever(user_id)
    vecstore_store_service = VecStoreService(llm_model_service, retriever, content, file_name)
    summarize_content = vecstore_store_service.save_docs_and_add_in_retriever()
    return summarize_content


@router.get("/")
async def get_answer(question: str, user_id: str):
    answer = await _invoke_agent(question, user_id)
    print("answer", answer)
    return answer


@router.post("/load_file")
async def load_file(file: UploadFile, user_id: str):
    destination = await _save_file(file)
    content = await _get_doc_content(destination)
    summarize_content = await _save_doc_content(content, user_id, file.filename)
    return summarize_content


@router.get("/my_files")
async def my_files(user_id: str):
    all_files_ids_names = DocumentsGetterService.get_files_ids_names(user_id)
    return all_files_ids_names


@router.get("/delete_all_files")
async def delete_all_files(user_id: str):
    VecStoreService.clear_vector_stores(user_id)
    DocumentsSaver.clear_user_directory(user_id)
    return "Загруженные документы удалены"


