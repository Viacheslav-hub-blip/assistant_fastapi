import os
from src.rag_agent_api.config import USERS_DIRECTORY
from langchain_core.documents import Document
import re


class DocumentsGetterService:
    @staticmethod
    def get_source_document(collection_name: str, doc_id: str, belongs_to: str, doc_number: str) -> Document:
        with open(
                rf"{USERS_DIRECTORY}\{collection_name}\{belongs_to}\{doc_id}_{doc_number}.txt",
                'r', encoding="utf-8") as f:
            content = f.readlines()
            doc = Document(page_content="".join(content))
        return doc

    @staticmethod
    def get_document_by_user_id_section_and_number(user_id: str, section: str, doc_number: str) -> Document:
        files = os.listdir(rf"{USERS_DIRECTORY}\user_{user_id}\{section}")
        for name in files:
            if name.split("_")[1].replace(".txt", "") == doc_number:
                with open(
                        rf"{USERS_DIRECTORY}\user_{user_id}\{section}\{name}",
                        'r', encoding="utf-8") as f:
                    content = f.readlines()
                    doc = Document(page_content="".join(content))
                    return doc
        return Document(page_content="")

    @staticmethod
    def get_files_ids_names(user_id: str) -> dict:

        path = rf'{USERS_DIRECTORY}\user_{user_id}\user_{user_id}_files'
        try:
            with open(path, 'r', encoding="utf-8") as f:
                content = f.readlines()
            ids_names = {id: name.rstrip() for id, name in [c.split("::") for c in content]}
            return ids_names
        except FileNotFoundError:
            return {"error": "File not found"}

    @staticmethod
    def get_files_summary(user_id: str) -> dict[str, str]:
        path = rf'{USERS_DIRECTORY}\user_{user_id}\user_{user_id}_files_summary'
        try:
            res = {}
            current_id = None
            with open(path, 'r', encoding="utf-8") as f:
                for line in f:
                    if '::' in line:
                        if current_id is not None:  # Сохраняем предыдущий текст, если есть
                            res[current_id] = current_text.strip()
                        current_id, current_text = line.split('::', maxsplit=1)
                    else:
                        if current_id is not None:
                            current_text += line
                    # Добавляем последнюю запись
                if current_id is not None:
                    res[current_id] = current_text.strip()
            return res
        except FileNotFoundError:
            return {"error": "File not found"}
