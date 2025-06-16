from typing import List, TypedDict, NamedTuple

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.rag_agent_api.prompts.rag_agent_prompts import (
    analyze_category_prompt,
    factual_query_chain_prompt,
    analytical_query_chain_prompt,
    opinion_query_chain_prompt,
    rerank_chain_prompt,
    check_possibilyty_response_prompt,
    answer_with_context_prompt,
    check_answer_for_correct_prompt,
    correct_answer_prompt
)
from src.rag_agent_api.services.database.documents_getter_service import DocumentsGetterService


class Message(NamedTuple):
    type: str
    message: str


class GraphState(TypedDict):
    question: str
    user_id: int
    workspace_id: int
    belongs_to: str
    chat_history: list[tuple[str, str]]

    question_category: str
    question_with_additions: str

    retrieved_documents: list[Document]
    neighboring_docs: list[Document]

    answer_with_retrieve: str
    answer: str
    answer_without_retrieve: bool

    used_docs: list[str]


class RagAgent:
    def __init__(self, model: BaseChatModel, retriever):
        self.model = model
        self.retriever = retriever
        self.state = GraphState
        self.app = self.compile_graph()

    def simple_chain(self, system_prompt: str, question: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Вопрос пользователя: {question}")
            ]
        )

        answer_chain = prompt | self.model | StrOutputParser()
        return answer_chain.invoke({"question": question})

    def analyze_query_for_category_chain(self, question: str) -> str:
        return self.simple_chain(analyze_category_prompt, question)

    def analyze_query_for_category(self, state: GraphState):
        """Анализирует вопрос и разделяет его на 3 категории:
        1. Фактическая
        2. Аналитическая
        3. Формирование мнения
        В зависимости от категории на следующих этапах убудут сформированы вспомогательные вопросы
        """
        question_category: str = self.analyze_query_for_category_chain(state["question"])
        return {"question": state["question"], "question_category": question_category}

    def choose_query_strategy(self, state: GraphState):
        """Взависимости от выбранной категории вопроса перенаправляет в цепочку"""
        query_strategy: str = state["question_category"].lower()

        if query_strategy == "factual":
            return "Factual"
        elif query_strategy == "analytical":
            return "Analytical"
        elif query_strategy == "opinion":
            return "Opinion"
        return "Simple"

    def factual_query_chain(self, question: str) -> str:
        return self.simple_chain(factual_query_chain_prompt, question)

    def factual_query_strategy(self, state: GraphState):
        """Цепочка, которая выполняется если выбран тип вопроса 'Фактический'
        В этом случае генерируются дполнительные воросы
        """
        return {"question_with_additions": self.factual_query_chain(state["question"])}

    def analytical_query_chain(self, question: str) -> str:
        return self.simple_chain(analytical_query_chain_prompt, question)

    def analytical_query_strategy(self, state: GraphState):
        """Цепочка которая выполняется в случае если выбран тип вопроса 'Аналитический'
        Для такого вопроса генерируются уточняющие вопросы
        """
        return {"question_with_additions": self.analytical_query_chain(state["question"])}

    def opinion_query_chain(self, question: str) -> str:
        return self.simple_chain(opinion_query_chain_prompt, question)

    def opinion_query_strategy(self, state: GraphState):
        """Цепочка которая выполняется в случае если выбран тип вопроса 'Формирование мнения'
        Для такого вопроса генерируются уточняющие вопросы
        """
        return {"question_with_additions": self.opinion_query_chain(state["question"])}

    def retrieve_documents(self, state: GraphState):
        """Ищет документы и ограничивает выборку документами со сходством <= 1.3(наиболее релевантные)"""
        retrieved_documents: List[Document] = self.retriever.get_relevant_documents(state["question_with_additions"],
                                                                                    state["belongs_to"],
                                                                                    )

        return {"retrieved_documents": retrieved_documents}

    def get_neighboring_numbers_doc(self, section_numbers_dict: dict) -> dict:
        """Получает словарь, где ключ - раздел документа, значение - номера документов в разделе
        Возвращает словарь, где к номерам документов добавляютс соседние номера кажого документа
        """
        res_dict = {}
        for sec, numbers in section_numbers_dict.items():
            numbers_int: list[int] = [int(s) for s in numbers.split("/")]
            neighboring_numbers: list[int] = numbers_int + [n - 1 for n in numbers_int] + [n + 1 for n in numbers_int]
            unique_neighboring_numbers = sorted(set(neighboring_numbers))
            res_dict[sec] = "/".join([str(i) for i in unique_neighboring_numbers])
        return res_dict

    def section_numbers_dict(self, retrieved_documents: list[Document]) -> dict:
        section_numbers_dict = {}
        for doc in retrieved_documents:
            if doc.metadata["belongs_to"] in section_numbers_dict:
                section_numbers_dict[doc.metadata["belongs_to"]] += f'/{doc.metadata["doc_number"]}'
            else:
                section_numbers_dict[doc.metadata["belongs_to"]] = str(doc.metadata["doc_number"])
        return section_numbers_dict

    def get_neighboring_docs(self, state: GraphState):
        """Ищет соседние исходные документы к тем, что были надйены при посике с помощью retriever"""
        section_numbers_dict = self.section_numbers_dict(state["retrieved_documents"])
        neighboring_docs_numbers: dict = self.get_neighboring_numbers_doc(section_numbers_dict)
        neighboring_docs: list[Document] = []
        for belongs_to, numbers in neighboring_docs_numbers.items():
            doc_nums = numbers.split("/")
            for num in doc_nums:
                chunk = DocumentsGetterService.get_source_chunk(state["user_id"], state["workspace_id"], belongs_to,
                                                                num)
                if len(chunk.page_content) > 0:
                    neighboring_docs.append(chunk)
        return {"neighboring_docs": neighboring_docs}

    def rerank_document_chain(self, question: str, document: Document) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", rerank_chain_prompt),
            ("human", "Вопрос пользователя: {question}")
        ])

        chain = prompt | self.model | StrOutputParser()
        return chain.invoke({"question": question, "document": document})

    def reranked_documents(self, state: GraphState):
        retrieved_neighboring_docs = state["neighboring_docs"]
        question = state["question"]
        docs_with_rank_over = []
        for doc in retrieved_neighboring_docs:
            rank = self.rerank_document_chain(question, doc)
            try:
                if int(rank) >= 3:
                    docs_with_rank_over.append(doc)
            except ValueError:
                print("Неправильная оценка", rank)
        return {"neighboring_docs": docs_with_rank_over}

    def checking_possibility_responses_chain(self, question: str, context: str):
        """Просим модель проверить, можно ли дать ответ на вопрос по найденным документам"""
        system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", check_possibilyty_response_prompt),
                ("human", "Вопрос: {question}")
            ]
        )
        chain = system_prompt | self.model | StrOutputParser()
        return chain.invoke({"question": question, "context": context})

    def answer_with_context_chain(self, question: str, context: str, chat_history: list[tuple[str, str]]):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", answer_with_context_prompt),
                MessagesPlaceholder("history"),
                ("human", "Вопрос: {question}")
            ]
        )

        chain = prompt | self.model | StrOutputParser()
        return chain.invoke({"history": chat_history, "question": question, "context": context})

    def generate_answer_with_retrieve_context(self, state: GraphState):
        doc_context = "".join([doc.page_content for doc in state["neighboring_docs"]])
        answer = self.answer_with_context_chain(state["question"], doc_context, state["chat_history"])
        return {"answer_with_retrieve": answer}

    def check_answer_for_correctness_chain(self, state: GraphState):

        system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", check_answer_for_correct_prompt),
                MessagesPlaceholder("history"),
                ("human", "Вопрос: {question}")
            ]
        )
        print(state["answer_with_retrieve"])
        chain = system_prompt | self.model | StrOutputParser()
        return chain.invoke(
            {"history": state["chat_history"],
             "retrieve_answer": state["answer_with_retrieve"],
             "question": state["question"]})

    def correct_answer_chain(self, question: str, answer: str, chat_history: list[tuple[str, str]]):
        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", correct_answer_prompt),
                MessagesPlaceholder("history"),
                ("human", "Вовпрос пользователя: {question}")
            ]
        )
        chain = final_prompt | self.model | StrOutputParser()
        return chain.invoke({"history": chat_history, "question": question, "answer": answer})

    def corrective_answer(self, state: GraphState):
        correct_answer = self.correct_answer_chain(
            state["question"],
            state["answer_with_retrieve"],
            state["chat_history"])
        return {"answer_with_retrieve": correct_answer}

    def final_answer(self, state: GraphState):
        return {"answer_without_retrieve": False, "answer":  state["answer_with_retrieve"]}

    def add_source_docs_names(self, state: GraphState):
        documents: list[Document] = state["neighboring_docs"]
        used_docs_names = list(set([doc.metadata["belongs_to"] for doc in documents]))
        return {"used_docs": used_docs_names}

    def compile_graph(self):
        workflow = StateGraph(self.state)
        workflow.add_node("analyze_query_for_category", self.analyze_query_for_category)
        workflow.add_node("factual_query_strategy", self.factual_query_strategy)
        workflow.add_node("analytical_query_strategy", self.analytical_query_strategy)
        workflow.add_node("opinion_query_strategy", self.opinion_query_strategy)
        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("get_neighboring_docs", self.get_neighboring_docs)
        workflow.add_node("reranked_documents", self.reranked_documents)
        workflow.add_node("generate_answer_with_retrieve_context", self.generate_answer_with_retrieve_context)
        workflow.add_node("final_answer", self.final_answer)
        workflow.add_node("corrective_answer", self.corrective_answer)
        workflow.add_node("add_source_docs_names", self.add_source_docs_names)

        workflow.add_edge(START, "analyze_query_for_category")
        workflow.add_conditional_edges(
            "analyze_query_for_category",
            self.choose_query_strategy,
            {
                "Factual": "factual_query_strategy",
                "Analytical": "analytical_query_strategy",
                "Opinion": "opinion_query_strategy",
            }
        )

        workflow.add_edge("factual_query_strategy", "retrieve_documents")
        workflow.add_edge("analytical_query_strategy", "retrieve_documents")
        workflow.add_edge("opinion_query_strategy", "retrieve_documents")

        workflow.add_edge("retrieve_documents", "get_neighboring_docs")
        workflow.add_edge("get_neighboring_docs", "reranked_documents")

        workflow.add_edge("reranked_documents", "generate_answer_with_retrieve_context")

        workflow.add_conditional_edges(
            "generate_answer_with_retrieve_context",
            self.check_answer_for_correctness_chain,
            {
                "БЕЗ ИЗМЕНЕНИЙ": "final_answer",
                "НУЖНЫ ИЗМЕНЕНИЯ": "corrective_answer"
            }
        )
        workflow.add_edge("corrective_answer", "final_answer")
        workflow.add_edge("final_answer", "add_source_docs_names")
        workflow.add_edge("add_source_docs_names", END)
        return workflow.compile()

    def __call__(self, *args, **kwargs):
        return self.app


if __name__ == "__main__":
    from src.rag_agent_api.services.retriever_service import CustomRetriever, embeddings
    from src.rag_agent_api.langchain_model_init import model_for_answer
    from langchain_chroma import Chroma
    from pprint import pprint

    vec_store = Chroma(
        collection_name="example_pro",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
        collection_metadata={"hnsw:space": "cosine"}
    )

    retriever = CustomRetriever(
        vec_store,
    )

    docs = [
        Document(
            page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
            metadata={"source": "tweet", "doc_id": "1"},
            id=1,
        ),
        Document(
            page_content="Slava Rylkov this is a young developer who is 20 years old. He lives in Moscow, studies at the university. SLava is interested in programming and language models.",
            metadata={"source": "tweet", "doc_id": "2"},
            id=2,
        ),
        Document(
            page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
            metadata={"source": "news", "doc_id": "3"},
            id=3),
        Document(
            page_content="The latest of  SLava project was the development of a financial platform and the creation of a smart assistant.",
            metadata={"source": "tweet", "doc_id": "4"},
            id=4,
        ),
        Document(
            page_content="Слава Рыльков - молодой разработчик, ему 20 лет. Он живет в Москве, учится в университете. Слава интересуется программированием и языковыми моделями.",
            metadata={"source": "tweet", "doc_id": "5"},
            id=5,
        ),
    ]
    retriever.vectorstore.add_documents(docs)

    agent = RagAgent(model_for_answer, retriever)

    while True:
        input_question = input("Введите сообщение: ")

        if input_question != "q":
            inputs = {"question": input_question}
            result = agent().invoke(inputs)
            print(result, result["forced_generation"])
            question, generation, web_search, forced_generation = result["question"], result["generation"], result[
                "web_search"], result["forced_generation"]
            try:
                documents: List[Document] = result["documents"]
            except:
                documents = []

            print("##QUESTION## ", question)
            print("##ANSWER## ", generation)
            print("##WEB_SERACH## ", web_search)
            pprint(documents)
            print()
        else:
            exit()
