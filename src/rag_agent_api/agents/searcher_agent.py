from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.rag_agent_api.agents.tools.search_agent_tool import search_tool, wiki_tool
from src.rag_agent_api.prompts.search_agent_prompts import (
    define_user_question_prompt,
generate_answer_prompt

)


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

    def define_user_question(self, state: SearcherState):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", define_user_question_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "Вопрос: {question}")
            ]
        )
        chain = prompt | self.model | StrOutputParser()
        answer = chain.invoke({"question": state["user_input"], "chat_history": state["chat_history"]})
        return {"user_input": answer}

    def search(self, state: SearcherState):
        web_search_result = search_tool.invoke(state["user_input"])["results"]
        wiki_search_result = wiki_tool.invoke(state["user_input"])
        searched_content = [r["content"] for r in web_search_result]
        return {"searched_content": " ".join(searched_content) + wiki_search_result}

    def generate_answer(self, state: SearcherState):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", generate_answer_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "Вопрос: {question}")
            ]
        )
        chain = prompt | self.model | StrOutputParser()
        answer = chain.invoke({"chat_history": state["chat_history"], "context": state["searched_content"],
                               "question": state["user_input"]})
        return {"answer": answer}

    def compile_graph(self):
        workflow = StateGraph(self.state)
        workflow.add_node("define_user_question", self.define_user_question)
        workflow.add_node("search", self.search)
        workflow.add_node("generate", self.generate_answer)

        workflow.add_edge(START, "define_user_question")
        workflow.add_edge("define_user_question", "search")
        workflow.add_edge("search", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def __call__(self, *args, **kwargs):
        return self.app
