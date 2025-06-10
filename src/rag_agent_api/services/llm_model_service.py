import time
from typing import NamedTuple, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.rag_agent_api.prompts.llm_model_service_prompts import (
    summarization_text_with_questions_prompt,
    summarization_with_max_word
)


class SummarizeContentAndDocs(NamedTuple):
    summary_texts: List[str]
    source_docs: List[str]


def exponential_backoff(retries, initial_delay=1):
    """Функция для расчета экспоненциальной задержки."""
    return min(initial_delay * (2 ** retries), 60)  # Максимальная задержка — 60 секунд


class LLMModelService:
    def __init__(self, model: BaseChatModel):
        self.model = model

    def _get_answer(self, prompt_text, documents: List[str]) -> SummarizeContentAndDocs:
        """Генерирует ответ модели по заданному prompt, который содержит поле element"""

        retries, max_retries, result_text_sum = 0, 5, []

        prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = {"element": lambda x: x} | prompt | self.model | StrOutputParser()
        batch_size = max(int(len(documents) * 0.2), 1)
        batches_split_docs = [documents[i: i + batch_size] for i in range(0, len(documents), batch_size)]

        for batch in batches_split_docs:
            while retries < max_retries:
                try:
                    text_sum = (summarize_chain
                                .with_retry(wait_exponential_jitter=True, stop_after_attempt=6)
                                .invoke(batch))
                    result_text_sum.append(text_sum)
                    retries = 0
                    break
                except Exception as e:
                    retries += 1
                    delay = exponential_backoff(retries)
                    time.sleep(delay)
        return SummarizeContentAndDocs(result_text_sum, documents)

    def get_summarize_docs_with_questions(self, split_docs: List[str]) -> SummarizeContentAndDocs:
        """Создает краткое описание к документам и добавлет вопросы к каждому фрагменту
        Возвращает краткие содержания и исходные фрагменты
        """
        return self._get_answer(summarization_text_with_questions_prompt, split_docs)

    def get_super_brief_content(self, content: str, max_word=70) -> str | Exception:
        """Вовзращает краткое содержание размера max_word"""
        try:
            chain = ChatPromptTemplate.from_template(summarization_with_max_word) | self.model | StrOutputParser()
            return chain.invoke({"max_word": max_word, "context": content})
        except Exception as e:
            return e
