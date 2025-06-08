from typing import TypedDict, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.rag_agent_api.agents.tools.visualizer_tools import table_creator


class VisualizerState(TypedDict):
    user_input: str
    chat_history: list[tuple[str, str]]
    answer_to_route: str
    route_tools: Literal["table", "piechart", "unknow"]
    answer: str
    isComplete: bool


class VisualizerAgent:
    def __init__(self, model: BaseChatModel):
        self.model = model
        self.state = VisualizerState
        self.app = self.compile_graph()

    def choose_tool(self, state: VisualizerState):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """
                Ты  - умный ассистент, который помогает пользователям визуализировать данные.
                
                Твоя задача:
                1. Проанализировать историю сообщений
                2. Проанализировать запрос пользователя 
                3. Выбрать один из доступных инструментов инструментов. Вот доступные тебе инструменты:
                
                table  - инструмент, который создает таблицы
                
                3. Вернуть одно слово - название инструмента или unknow, если ни один инструмент не подходит
                
                Верни только одно слово из table, unknow          
                
                """),
                MessagesPlaceholder("history"),
                ("human", "{question}")
            ]
        )

        chain = prompt | self.model | StrOutputParser()
        ans = chain.invoke({"history": state["chat_history"], "question": state["user_input"]})
        return {"answer_to_route": ans}

    def route_tools(self, state: VisualizerState) -> Literal["table", "piechart", "unknow"]:
        selected_tool = state["answer_to_route"]

        if "table" in selected_tool:
            return "table"
        return "unknow"

    def handle_table_creator(self, state: VisualizerState):
        llm_with_tools = self.model.bind_tools([table_creator])
        examples = [
            AIMessage(
                content="Температура: 23.5, 18.2, 25.0 Влажность: 65, 72, 58 Давление: 101, 100, 101"
            ),
            HumanMessage(
                "Представь в иде таблицы"
            ),
            AIMessage(
                "",
                name="example_assistant",
                tool_calls=[
                    {"name": "table creator tool",
                     "args": {"index": "0, 1, 2", "columns_names": "Температура, Влажность, Давление",
                              "data": "23.5, 18.2, 25.0; 65, 72, 58; 101, 100, 101"},
                     "id": "1"
                     },
                ]
            ),
            ToolMessage("", tool_call_id="1"),

            HumanMessage(
                """
                Построй таблицу по моим данные о ценах на продукты

                -100 молоко
                -130 хлеб
                -50 чипсы
                """
            ),
            AIMessage(
                "",
                name="example_assistant",
                tool_calls=[
                    {"name": "table creator tool",
                     "args": {"index": "0, 1, 2", "columns_names": "продукт, цена",
                              "data": "молоко, 100; хлеб, 130; чипсы, 50;"},
                     "id": "2"
                     },
                ]
            ),
            ToolMessage("", tool_call_id="2")

        ]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """
                Ты  - умный ассистент, который помогает пользователям визулизировать данные с помощью таблиц. 
                Ниже приведены примеры использования инструента и история сообщений с пользователем. 
                Твоя задача: выдели данные из истории собщений для построение таблицы.
                Примеры:
                """),
                *examples,
                MessagesPlaceholder("history"),
                ("human", "{question}")
            ]
        )

        chain = prompt | llm_with_tools
        answer = chain.invoke({"history": state["chat_history"], "question": state["user_input"]})

        for tool_call in answer.tool_calls:
            try:
                output = table_creator.invoke(tool_call["args"])
                return {"isComplete": True, "answer": output}
            except Exception as e:
                print("НЕ УДАЛОСЬ ПОСТРОИТЬ ТАБЛИЦУ", e)
                return {"isComplete": False, "answer": str(e)}

    def handle_unknow(self, state: VisualizerState):
        return {"isComplete": False, "answer": "не удалось выбрать инструмент"}

    def compile_graph(self):
        workflow = StateGraph(self.state)
        workflow.add_node("choose_tool", self.choose_tool)
        workflow.add_node("handle_table_creator", self.handle_table_creator)
        workflow.add_node("handle_unknow", self.handle_unknow)

        workflow.add_edge(START, "choose_tool")
        workflow.add_conditional_edges(
            "choose_tool",
            self.route_tools,
            {
                "table": "handle_table_creator",
                "unknow": "handle_unknow"
            }
        )
        workflow.add_edge("handle_table_creator", END)
        workflow.add_edge("handle_unknow", END)

        return workflow.compile()

    def __call__(self, *args, **kwargs):
        return self.app
