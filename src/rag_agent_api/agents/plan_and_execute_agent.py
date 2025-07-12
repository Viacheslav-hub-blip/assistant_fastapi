import json
from pprint import pprint
from typing import Any, NamedTuple

from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import tool, InjectedToolArg
from typing_extensions import Annotated

from src.rag_agent_api.agents.rag_agent import RagAgent
from src.rag_agent_api.agents.searcher_agent import SeracherAgent
from src.rag_agent_api.langchain_model_init import model_for_answer
from langchain_core.documents import Document

class RagSearchRes(NamedTuple):
    answer: str
    used_docs: list[str]
    neighboring_docs: list[str]


class PlanResult(NamedTuple):
    answer: str
    used_docs: list[str]
    neighboring_docs: list[Document]


@tool
def web_search(question: str) -> str:
    """Используй этот иннструмент для ответа на вопрос с помощью поиска в интернете"""

    searcher = SeracherAgent(model_for_answer)
    answer = searcher().invoke({"user_input": question})["answer"]
    return answer


@tool
def rag_search(
        question: str,
        user_id: Annotated[int, InjectedToolArg],
        workspace_id: Annotated[int, InjectedToolArg],
        belongs_to: Annotated[str, InjectedToolArg],
        chat_history: Annotated[list[tuple[str, str]], InjectedToolArg],
        retriever: Annotated[Any, InjectedToolArg],

) -> RagSearchRes:
    """Поиск в векторном хранилище пользователя"""
    rag_agent = RagAgent(model_for_answer, retriever)
    result = rag_agent().invoke(
        {"question": question,
         "user_id": user_id,
         "workspace_id": workspace_id,
         "belongs_to": belongs_to,
         "chat_history": chat_history}
    )

    return RagSearchRes(result["answer"], result["used_docs"], result["neighboring_docs"])


tools = [web_search, rag_search]


class PlanAndExecuteAgent:
    def __init__(self,
                 llm: LanguageModelLike,
                 max_plan_length,
                 user_id: int,
                 workspace_id: int,
                 belongs_to: str,
                 retriever,
                 chat_history: list[tuple[str, str]],
                 ):
        self.llm = llm
        self.max_plan_length = max_plan_length
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.belongs_to = belongs_to
        self.retriever = retriever
        self.chat_history = chat_history

        self.tools = {t.name: t for t in tools}
        self.plan = []
        self.current_step = 0
        self.result_steps = {}

        self.used_docs = []
        self.neighboring_docs = []

    def run(self, task: str) -> PlanResult:
        self.plan = self._create_plan(task, self.chat_history)
        self.current_step = 0

        for step in self.plan:
            result = self._execute_tool(step, self.result_steps)
            self.result_steps[step] = result
            if isinstance(result, str) and result.startswith("Ошибка"):
                print("ошибка выполнения плана, переплан")
                self.result_steps = {}
                return self._replan(task, result)
        return self._final_result()

    def _create_plan(self, task, chat_history) -> list[str]:
        prompt = f"""
        Ты  - умный ассистент, который разбивает запрос пользователя на отдельные шаги.
        
        История диалога с пользователем:
        {chat_history}
        
        Каждый шаг должен быть конкретным и выполнимым с помощью одного из доступных инструментов или с помощью собственных знаний:
        {', '.join(self.tools.keys())}
        
        Инструкции:
        -проанализируй запрос пользователя 
        -всегда начинай с вызова интсрумента rag_search
        -при необходимо использовать актуальные на данный момент данные используй инструмент web_search

        Твоя задача: Разбей следующую задачу на последовательность шагов (не более {self.max_plan_length})
      
        Задача: {task}

        
         Верни ответ в формате JSON:
        {{
            "plan": [
                "шаг 1",
                "шаг 2",
                ...
            ]
        }}

        Отвечай только в формате JSON. Проверь валидность ответа. Размышляй шаг за шагом. 
        Всегда начинай с использования rag_search. 
        """
        response = self.llm.invoke(prompt)
        pprint(response.content)
        try:
            plan = json.loads(response.content)["plan"]
            return plan
        except:
            return [f"INVALID_PLAN: {response}"]

    def _execute_tool(self, step: str, previous_steps: dict) -> str:
        prompt = f"""
        Ты  - умный ассистент, который выполняет шаги плана действий. 

        Предыдущие шаги и результаты выполнения:
        {previous_steps}

        Текущий шаг плана: {step}

        Определи, какой инструмент подходит для выполнения этого шага и какие входные данные ему нужны.
        Доступные инструменты:
        {json.dumps({name: t.args for name, t in self.tools.items()})}

        Верни ответ в формате JSON:
        {{
            "thought": "рассуждения о выборе инструмента",
            "action": "имя инструмента",
            "action_input": "входные данные"
        }}       

        Отвечай строго в формате JSON. Проверь валидность ответа. 
        """
        pprint(f"prompt для выбора инструмента {prompt}")
        response = self.llm.invoke(prompt)
        pprint(f"Аргументы для инстурмента {response.content}")
        try:
            data = json.loads(response.content)
            selected_tool = self.tools[data["action"]]
            if not selected_tool:
                return f"Ошибка: инструмент {data['action']} не найден"

            if selected_tool.name == "rag_search":
                answer = tools.invoke(
                    data["action_input"],
                    self.user_id,
                    self.workspace_id,
                    self.belongs_to,
                    self.chat_history,
                    self.retriever
                )
                self.used_docs = answer.used_docs
                self.neighboring_docs = answer.neighboring_docs
                print("RAG SEARCH ANSWER", answer)
                return answer.answer
            return selected_tool.invoke(data["action_input"])
        except Exception as e:
            return f"Ошибка выполнения шага: {str(e)}"

    def _replan(self, task: str, error: str) -> PlanResult:
        prompt = f"""
        При выполнении задачи возникла ошибка:
        {error}

        Исходная задача: {task}
        Первоначальный план:
        {json.dumps(self.plan, indent=2)}

        Создай новый план, учитывая возникшую ошибку.
        Верни ответ в том же JSON формате.
        """
        self.plan = self._create_plan(prompt, self.chat_history)
        return self.run(task)

    def _final_result(self) -> PlanResult:
        prompt = f"""
            Ты - умный ассистент который отвечает на запросы пользователя. 
            История диалога с пользователем:
            {self.chat_history}
        
              Все шаги плана были успешно выполнены:
              {json.dumps(self.plan, indent=2)}

              Результаты выполнения:
              {self.result_steps}
              
              Формат вывода:
              -отвечай строго в формате Markdown
              
              
                Используй следующие элементы для ответа в формате Markdown :
                - Заголовки (`##`, `###`)
                - **Жирный текст**, *курсив*
                - Списки (`-`, `1.`)
                - Блоки кода (```python\n...```)
                - Формулы LaTeX (`$$E=mc^2$$`)
                - Mermaid-диаграммы (```mermaid\ngraph TD\n...```)

              Проанализируй результаты выполнения и сформулируй итоговый ответ на исходную задачу.
              Не упоминай результаты выполнения предыдущих шагов и поставновку задачи, верни только финальный ответ. 
              Проверь валидность ответа в Markdown. 
              """

        answer = self.llm.invoke(prompt)
        return PlanResult(answer.content, self.used_docs, self.neighboring_docs)


if __name__ == '__main__':
    agent = PlanAndExecuteAgent(model_for_answer, 3)
    print(agent.run("сколько лететь до марса на космическом корабле?"))
