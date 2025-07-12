from typing import TypedDict, Literal, Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Command

from src.rag_agent_api.agents.rag_agent import RagAgent
from src.rag_agent_api.agents.searcher_agent import SeracherAgent
from src.rag_agent_api.agents.visualizer_agent import VisualizerAgent
from src.rag_agent_api.prompts.supervisor_prompts import routing_prompt, simple_task_prompt


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
    used_docs_names: list[str]


class SuperVisor:
    def __init__(self, model: BaseChatModel, retriever):
        self.model = model
        self.retriever = retriever
        self.state = SupervisorState
        self.app = self.compile_graph()

    def route_task(self, state: SupervisorState):
        print("ROUTE TASK")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", routing_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "Вопрос: {question}")
            ]
        )
        chain = prompt | self.model | StrOutputParser()
        ans = chain.invoke({"chat_history": state["chat_history"], "question": state["user_input"]})
        if ans == "visualizer":
            return Command(goto="visualizer", update={"routing": ans})
        if ans == "rag_agent":
            return Command(goto="rag_agent", update={"routing": ans})
        if ans == "web_searcher":
            return Command(goto="web_searcher", update={"routing": ans})
        return Command(goto="simple", update={"routing": ans})

    def handle_rag_agent(self, state: SupervisorState):
        print("rag agent")
        rag_agent = RagAgent(self.model, self.retriever)
        result = rag_agent().invoke(
            {"question": state["user_input"],
             "user_id": state["user_id"],
             "workspace_id": state["workspace_id"],
             "belongs_to": state["belongs_to"],
             "chat_history": state["chat_history"]}
        )
        print("ANSWER RAG AGENT", result)
        agent_result = {
            "type": "rag_agent"
        }
        return {
            "agent_result": agent_result,
            "complete": True,
            "answer": result.get("answer", ""),
            "neighboring_docs": [doc.page_content for doc in result["neighboring_docs"]],
            "used_docs_names": result["used_docs"]
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

    def web_searcher(self, state: SupervisorState):
        print("web searcher")
        searcher_agent = SeracherAgent(self.model)

        result = searcher_agent().invoke(
            {"chat_history": state["chat_history"],
             "user_input": state["user_input"]}
        )

        agent_result = {
            "type": "visualizer",
            "data": result,
            "complete": True,
        }

        return {
            "agent_result": agent_result,
            "complete": agent_result["complete"],
            "answer": result.get("searched_content", "Не удалось дать овтет")
        }

    def handle_simple_task(self, state: SupervisorState):
        print("simple task")

        chain = ChatPromptTemplate.from_template(simple_task_prompt) | self.model | StrOutputParser()
        answer = chain.invoke({"chat_history": state["chat_history"], "user_input": state["user_input"]})
        agent_result = {
            "type": "simple",
        }
        return {
            "agent_result": agent_result,
            "complete": True,
            "answer": answer,
        }

    def supervisor_results(self, state: SupervisorState):
        result = state["agent_result"]
        print("AGENT TYPE", result["type"])
        print("================STATE RESULT=====================")
        print(state)
        match result["type"]:
            case "rag_agent":
                return {"complete": True}
            case "visualizer":
                if not result["complete"]:
                    print("Visualizer не завершил работу. Требуется дополнительная обработка.")
                    return {"complete": False}
                return {"use_visualizer": True, "complete": True}
            case "web_searcher":
                return {"complete": True}
            case "simple":
                return {"complete": True}
            case _:
                print(f"Неизвестный тип результата: {result['type']}")
                return {"complete": False}

    def compile_graph(self):
        workflow = StateGraph(self.state)
        workflow.add_node("route_task", self.route_task)
        workflow.add_node("rag_agent", self.handle_rag_agent)
        workflow.add_node("simple", self.handle_simple_task)
        workflow.add_node("visualizer", self.handle_visualizer_task)
        workflow.add_node("supervisor", self.supervisor_results)
        workflow.add_node("web_searcher", self.web_searcher)

        workflow.add_edge(START, "route_task")

        workflow.add_edge("rag_agent", "supervisor")
        workflow.add_edge("visualizer", "supervisor")
        workflow.add_edge("simple", "supervisor")
        workflow.add_edge("web_searcher", "supervisor")

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
