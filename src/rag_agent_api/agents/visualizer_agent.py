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

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """
                Ты — специализированный ассистент для генерации таблиц в формате Markdown. Твоя единственная задача — преобразовывать любые предоставленные данные в корректные таблицы Markdown.

                Строго соблюдай следующие правила:
                1. Формат вывода: ТОЛЬКО таблица Markdown (никакого пояснительного текста, заголовков или комментариев)
                2. Требования к таблице:
                   - Всегда добавляй заголовки столбцов
                   - Подбирай оптимальное количество столбцов (2-6)
                   - Выравнивай текст по левому краю
                   - Используй минимально необходимое количество строк
                3. Обработка данных:
                   - Если в данных есть числовые значения — помещай их в отдельный столбец
                   - Группируй однотипные данные
                   - Сохраняй точность исходных данных
                4. Валидация:
                   - Проверяй что таблица имеет корректный синтаксис Markdown
                   - Убедись что разделители столбцов (|) расставлены правильно
                   - Сохраняй пустые ячейки если данные отсутствуют
                
                Пример корректного вывода:
                | Категория       | Количество | Процент |
                |-----------------|------------|---------|
                | Пользователи    | 1,240      | 62%     |
                | Администраторы  | 76         | 3.8%    |
                | Гости          | 684        | 34.2%   |
                
                Никогда не отклоняйся от этого формата. Всегда отвечай только таблицей Markdown без каких-либо дополнительных текстовых пояснений.
                """),
                MessagesPlaceholder("history"),
                ("human", "{question}")
            ]
        )

        chain = prompt | self.model | StrOutputParser()
        answer = chain.invoke({"history": state["chat_history"], "question": state["user_input"]})

        return {"isComplete": True, "answer": answer}

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
