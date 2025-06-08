from typing import TypedDict, Literal, Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.rag_agent_api.agents.rag_agent import RagAgent
from src.rag_agent_api.agents.searcher_agent import SeracherAgent
from src.rag_agent_api.agents.visualizer_agent import VisualizerAgent


class SupervisorState(TypedDict):
    user_input: str
    answer: str
    use_web_search: bool
    use_visualizer: bool

    routing: Literal["retriever", "visualizer"]
    complete: bool
    agent_result: dict[str, Any]  # Новое поле для хранения результатов агентов
    chat_history: list[tuple[str, str]]

    # RAG AGENT STATE
    user_id: int
    workspace_id: int
    belongs_to: str

    question_category: str
    question_with_additions: str

    retrieved_documents: list[Document]
    neighboring_docs: list[Document]
    used_docs: list[str]


class SuperVisor:
    def __init__(self, model: BaseChatModel, retriever):
        self.model = model
        self.retriever = retriever
        self.state = SupervisorState
        self.app = self.compile_graph()

    def route_task(self, state: SupervisorState):
        system_prompt = """
        Ты  - полезный помощник, который управляет командной специализированных агентов.
        
        В вашем распоряжении два специализированных агента:        
        1. Агент по работе с нформацией (Retriever). Он извлекает ифнормацию из пользовательских документов
        2. Агент для визуализации информации (Visualizer). Он строит диаграммы, графики и другие графические элементы
        
        
        Ваша задача:
        1. Проанализировать запрос пользователя
        2. Опрделеить, какой специализированный агент должен проанализировать запрос 
        3. Вернуть одно слово: название агента - retriever, visualizer
        
        Для ответа используй только слова: retriever, visualizer
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )

        chain = prompt | self.model | StrOutputParser()

        ans = chain.invoke({"question": state["user_input"]})
        return {"routing": ans}

    def route_to_next_step(self, state: SupervisorState) -> Literal["retriever", "visualizer"]:
        route = state["routing"].lower()
        if route == "visualizer":
            return "visualizer"
        return "retriever"

    def handle_retriever_task(self, state: SupervisorState):
        rag_agent = RagAgent(
            self.model,
            self.retriever,
        )

        result = rag_agent().invoke(
            {"question": state["user_input"],
             "user_id": state["user_id"],
             "workspace_id": state["workspace_id"],
             "belongs_to": state["belongs_to"],
             "chat_history": state["chat_history"]}
        )

        agent_result = {
            "type": "retriever",
            "data": result,
            "complete": bool(result.get("used_docs", None)),
        }

        return {
            "agent_result": agent_result,
            "complete": agent_result["complete"],
            "answer": result.get("answer", ""),
            "used_docs": result.get("used_docs", []),
            "neighboring_docs": [doc.page_content for doc in result.get("neighboring_docs", [])],
            "used_docs_names": result.get("used_docs", [])
        }

    def handle_visualizer_task(self, state: SupervisorState):
        visualizer_agent = VisualizerAgent(self.model)

        result = visualizer_agent().invoke(
            {"chat_history": state["chat_history"],
             "user_input": state["user_input"]}
        )

        agent_result = {
            "type": "visualizer",
            "data": result,
            "complete": result["isComplete"],
        }

        return {
            "agent_result": agent_result,
            "complete": agent_result["complete"],
            "answer": result.get("answer", "")
        }

    def supervisor_results(self, state: SupervisorState):
        result = state["agent_result"]

        match result["type"]:
            case "retriever":
                if not result["complete"]:
                    print("Retriever не нашел документы")
                    searcher = SeracherAgent(self.model)
                    try:
                        answer = searcher().invoke({"user_input": state["user_input"]})["answer"]
                        return {"complete": True, "answer": answer, "use_web_search": True}
                    except Exception as e:
                        return {"complete": False}
                return {"complete": True}

            case "visualizer":
                if not result["complete"]:
                    print("Visualizer не завершил работу. Требуется дополнительная обработка.")
                    return {"complete": False}
                return {"use_visualizer": True, "complete": True}

            case _:
                print(f"Неизвестный тип результата: {result['type']}")
                return {"complete": False}

    def compile_graph(self):
        workflow = StateGraph(self.state)
        workflow.add_node("route_task", self.route_task)
        workflow.add_node("retriever", self.handle_retriever_task)
        workflow.add_node("visualizer", self.handle_visualizer_task)
        workflow.add_node("supervisor", self.supervisor_results)

        workflow.add_edge(START, "route_task")
        workflow.add_conditional_edges(
            "route_task",
            self.route_to_next_step,
            {
                "retriever": "retriever",
                "visualizer": "visualizer"
            }
        )
        workflow.add_edge("retriever", "supervisor")
        workflow.add_edge("visualizer", "supervisor")

        workflow.add_edge("supervisor", END)

        return workflow.compile()

    def __call__(self, *args, **kwargs):
        return self.app


if __name__ == "__main__":
    from src.rag_agent_api.langchain_model_init import model_for_answer
    from src.rag_agent_api.services.retriever_service import VectorDBManager

    question = "  кто такой илон маск"

    retriever = VectorDBManager.get_or_create_retriever(10, 20)
    super_visor = SuperVisor(
        model=model_for_answer,
        retriever=retriever,
    )
    # chat_history = [
    #     ("user", "сколько субьектов в России?"),
    #     ("assistant", """
    #     В России 89 субъектов федерации. Из них:
    #
    #     - 48 областей
    #     - 24 республики
    #     - 9 краев
    #     - 3 города федерального значения
    #     - 4 автономных округа
    #     - 1 автономная область
    #
    #     Это указано в представленном контексте: "В состав Российской Федерации входят 89 субъектов,
    #     48 из которых именуются областями, 24 — республиками, 9 — краями, 3 — городами федерального значения,
    #     4 — автономными округами и 1 — автономная область
    #     """),
    #     ("user", "построй таблицу по этим данным")
    # ]
    chat_history = [
        ("user", """
        кто такой илон маск
        """)
    ]

    result = super_visor().invoke({"user_input": question, "user_id": 10, "workspace_id": 20, "belongs_to": None,
                                   "chat_history": chat_history})

    print("RESULT", result)
    print("ANSWER", result["answer"])
