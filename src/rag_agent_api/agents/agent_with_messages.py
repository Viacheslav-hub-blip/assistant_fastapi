from langgraph.graph import END, START, StateGraph
from typing import TypedDict, Annotated
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    question: str
    final_answer: str


class AgentWithHistory:
    def __init__(self, model: BaseChatModel):
        self.model = model
        self.state = GraphState
        self.app = self.compile_graph()

    def call_model(self, state: GraphState):
        summary = state.get("summary", "")
        if summary:
            system_message = f'''Ты умный помощник,который отвечает на вопросы пользователя.
                                У тебя есть история диалога и вопрос пользователя. Твоя задача:
                                Ипользуя историю и свои знания ответить на вопрос пользователя, не используй специальных символов, например ### или ** и другие\n
                                Краткое содержание истории диалога:\n
                                {summary}'''
            messages = [SystemMessage(content=system_message)] + state["messages"]
        else:
            messages = state["messages"]

        response = self.model.invoke(messages)
        return {"messages": response}

    def summarize_conversation(self, state: GraphState):
        summary = state.get("summary", "")

        if summary:
            summary_message = (
                f"Вот краткая исотрия история диалога с пользователем: {summary}\n\n"
                f"Дополни эту историю диалога приняв во внимание следующие сообщения:"
            )
        else:
            summary_message = "Создай историю краткую историю диалога из сообщений выше"

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = self.model.invoke(messages)

        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}

    def should_continue(self, state: GraphState):
        messages = state["messages"]

        if len(messages) > 6:
            return "summarize_conversation"
        return "conversation"

    def format_answer(self, state: GraphState):
        answer: AIMessage = state["messages"][-1]
        answer_format = answer.content.replace("**", "")
        answer_format = answer_format.replace("***", "")
        answer_format = answer_format.replace("###", "")
        return {"final_answer": answer_format}

    def __call__(self, *args, **kwargs):
        return self.app

    def compile_graph(self):
        workflow = StateGraph(self.state)
        workflow.add_node("conversation", self.call_model)
        workflow.add_node("summarize_conversation", self.summarize_conversation)
        workflow.add_node("format_answer", self.format_answer)

        workflow.add_conditional_edges(START, self.should_continue)
        workflow.add_edge("summarize_conversation", "conversation")
        workflow.add_edge("conversation", "format_answer")
        workflow.add_edge("format_answer", END)

        return workflow.compile()


