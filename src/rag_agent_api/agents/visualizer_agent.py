from typing import TypedDict, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.rag_agent_api.prompts.visualizer_agent_prompts import (
    choose_tool_prompt,
    table_create_prompt
)


class VisualizerState(TypedDict):
    user_input: str
    chat_history: list[tuple[str, str]]
    answer_to_route: str
    route_tools: Literal["table", "piechart", "unknow"]
    answer: str
    isComplete: bool


def _prompt_creator(system_prompt) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{question}")
        ]
    )


class VisualizerAgent:
    def __init__(self, model: BaseChatModel):
        self.model = model
        self.state = VisualizerState
        self.app = self.compile_graph()

    def choose_tool(self, state: VisualizerState):
        chain = _prompt_creator(choose_tool_prompt) | self.model | StrOutputParser()
        ans = chain.invoke({"history": state["chat_history"], "question": state["user_input"]})
        return {"answer_to_route": ans}

    def route_tools(self, state: VisualizerState) -> Literal["table", "piechart", "unknow"]:
        if "table" in state["answer_to_route"]:
            return "table"
        return "unknow"

    def handle_table_creator(self, state: VisualizerState):
        chain = _prompt_creator(table_create_prompt) | self.model | StrOutputParser()
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
