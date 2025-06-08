from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.rag_agent_api.agents.tools.search_agent_tool import search_tool


class SearcherState(TypedDict):
    user_input: str
    chat_history: list[tuple[str, str]]
    answer: str
    searched_content: str


class SeracherAgent:
    def __init__(self, model):
        self.model = model
        self.state = SearcherState
        self.app = self.compile_graph()

    def search(self, state: SearcherState):
        search_result = search_tool.invoke(state["user_input"])
        searched_content = [r["content"] for r in search_result]
        return {"searched_content": " ".join(searched_content)}

    def generate_answer(self, state: SearcherState):
        sys_prompt = """
        Ты  - умный ассистент, который отвечает на вопросы пользователя используя найденный контекст строго в формате Markdown.
        Ты должен отвечать строго в формате Markdown. 
        
        Используй следующие элементы для ответа в формате Markdown :
        - Заголовки (`##`, `###`)
        - **Жирный текст**, *курсив*
        - Списки (`-`, `1.`)
        - Блоки кода (```python\n...```)
        - Таблицы (`| Столбец | Описание |`)
        - Формулы LaTeX (`$$E=mc^2$$`)
        - Mermaid-диаграммы (```mermaid\ngraph TD\n...```)
        
        Найденный контекст:
        {context}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("human", "{user_input}")
        ])
        chain = prompt | self.model | StrOutputParser()
        answer = chain.invoke({"context": state["searched_content"], "user_input": state["user_input"]})
        return {"answer": answer}

    def compile_graph(self):
        workflow = StateGraph(self.state)
        workflow.add_node("search", self.search)
        workflow.add_node("generate", self.generate_answer)

        workflow.add_edge(START, "search")
        workflow.add_edge("search", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def __call__(self, *args, **kwargs):
        return self.app
