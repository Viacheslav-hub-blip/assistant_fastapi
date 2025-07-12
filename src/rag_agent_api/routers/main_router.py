from pprint import pprint
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
    answer_id: int | None
    answer: str
    use_web_search: bool
    use_visualizer: bool
    used_docs_names: List[str]
    used_docs: List[str]


async def format_agent_answer(answer) -> AgentAnswer:
    used_docs_names, used_docs = [], []
    use_web_search, use_visualizer = False, False
    question, generation = answer["user_input"], answer["answer"]
    if answer.get("used_docs_names", None):
        used_docs_names = answer["used_docs_names"]
        used_docs = answer["neighboring_docs"]
    else:
        if answer.get("use_web_search", None):
            use_web_search = True
        if answer.get("use_visualizer", None):
            use_visualizer = True

    return AgentAnswer(None, generation, use_web_search, use_visualizer, used_docs_names, used_docs)


async def _invoke_agent(question: str, user_id: int, workspace_id: int, belongs_to: str,
                        chat_history: list[Message]) -> AgentAnswer:
    retriever = VectorDBManager.get_or_create_retriever(user_id, workspace_id)
    super_visor = SuperVisor(model=model_for_answer, retriever=retriever)
    chat_history = [(mess.type, mess.message) for mess in chat_history][:5]
    print("invoke", question, user_id, workspace_id, belongs_to, chat_history)
    try:
        result = super_visor().invoke(
            {"user_input": question, "user_id": user_id, "workspace_id": workspace_id, "belongs_to": belongs_to,
             "chat_history": chat_history})
        print("VISOR RESULT", result)
        return await format_agent_answer(result)
    except Exception as e:
        print("ОШИБКА ОБРАБОТКИ ЗАПРОСА", e)
        return await format_agent_answer({"user_input": question, "answer": "произошла ошибка"})


@router.get("/")
async def get_answer(question: str, user_id: int, workspace_id: int, belongs_to: str = None) -> dict[str, Any]:
    print(question, user_id, workspace_id, belongs_to)
    MessagesService.insert_message(user_id, workspace_id, question, "user")
    chat_history = MessagesService.get_user_messages(user_id, workspace_id)
    belongs_to = belongs_to if belongs_to != 'null' else None
    answer = await _invoke_agent(question, user_id, workspace_id, belongs_to, chat_history)
    answer._replace(answer_id=MessagesService.insert_message(user_id, workspace_id, answer.answer, "assistant"))
    print("=" * 50)
    print("=" * 50)
    print("=" * 50)
    pprint(answer)
    return answer._asdict()


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


@router.post("/favorite_message")
async def favorite_message(id: int, user_id: int, workspace_id: int, message: str) -> None:
    MessagesService.add_in_favorite(id, user_id, workspace_id, message)
    MessagesService.update_favorite_status_in_history(id, user_id, workspace_id, True)


@router.post("/unfavorite_message")
async def unfavorite_message(id: int, user_id: int, workspace_id: int) -> None:
    MessagesService.delete_from_favorites(id, user_id, workspace_id)
    try:
        MessagesService.update_favorite_status_in_history(id, user_id, workspace_id, False)
    except:
        pass


@router.get("/all_favorite_messages")
async def all_favorite_messages(user_id: int) -> list[dict[str, Any]]:
    return [mes._asdict() for mes in MessagesService.select_all_favorite_messages(user_id)]
