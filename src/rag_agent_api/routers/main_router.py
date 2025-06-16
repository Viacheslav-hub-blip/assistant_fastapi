from typing import NamedTuple, List, Any

# FastApi
from fastapi import APIRouter

from src.rag_agent_api.agents.supervisor_agent import SuperVisor
# AGENTS
from src.rag_agent_api.langchain_model_init import model_for_answer
from src.rag_agent_api.services.database.messages_service import MessagesService, Message
# SERVICES
from src.rag_agent_api.services.retriever_service import VectorDBManager

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)


class AgentAnswer(NamedTuple):
    answer: str
    use_web_search: bool
    use_visualizer: bool
    used_docs_names: List[str]
    used_docs: List[str]


async def format_agent_answer(answer) -> AgentAnswer:
    used_docs_names, used_docs = [], []
    use_web_search, use_visualizer = False, False
    question, generation = answer["user_input"], answer["answer"]

    if answer.get("used_docs", None):
        used_docs_names = answer["used_docs"]
        used_docs = answer["neighboring_docs"]
    else:
        if answer.get("use_web_search", None):
            use_web_search = True
        if answer.get("use_visualizer", None):
            use_visualizer = True

    return AgentAnswer(generation, use_web_search, use_visualizer, used_docs_names, used_docs)


async def _invoke_agent(question: str, user_id: int, workspace_id: int, belongs_to: str,
                        chat_history: list[Message]) -> AgentAnswer:
    retriever = VectorDBManager.get_or_create_retriever(user_id, workspace_id)
    super_visor = SuperVisor(model=model_for_answer, retriever=retriever)
    chat_history = [(mess.type, mess.message) for mess in chat_history]

    try:
        result = super_visor().invoke(
            {"user_input": question, "user_id": user_id, "workspace_id": workspace_id, "belongs_to": belongs_to,
             "chat_history": chat_history})
        return await format_agent_answer(result)
    except Exception as e:
        print("ОШИБКА ОБРАБОТКИ ЗАПРОСА", e)
        return await format_agent_answer({"user_input": question, "answer": "произошла ошибка"})


@router.get("/")
async def get_answer(question: str, user_id: int, workspace_id: int, belongs_to: str = None) -> AgentAnswer:
    MessagesService.insert_message(user_id, workspace_id, question, "user")
    chat_history = MessagesService.get_user_messages(user_id, workspace_id)
    belongs_to = belongs_to if belongs_to != 'null' else None
    answer = await _invoke_agent(question, user_id, workspace_id, belongs_to, chat_history)
    MessagesService.insert_message(user_id, workspace_id, answer.answer, "assistant")
    return answer


@router.get("/clear_chat_history")
async def clear_chat_history(user_id: int, workspace_id: int) -> dict[str, int]:
    MessagesService.delete_messages(user_id, workspace_id)
    return {"status": 200}


@router.get("/get_messages")
async def get_user_messages(user_id: int, workspace_id: int) -> list[dict[str, Any]]:
    messages = MessagesService.get_user_messages(user_id, workspace_id)
    print(messages)
    return [message._asdict() for message in messages]


@router.get("/save_message")
async def save_message(user_id: int, workspace_id: int, message: str, type: str) -> None:
    MessagesService.insert_message(user_id, workspace_id, message, type)
