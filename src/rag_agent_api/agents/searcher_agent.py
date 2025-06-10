from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from src.rag_agent_api.prompts.search_agent_prompts import generate_answer_prompt
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

        prompt = ChatPromptTemplate.from_messages([
            ("system", generate_answer_prompt),
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
