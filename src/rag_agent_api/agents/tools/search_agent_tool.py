import os

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch

tavily_api_key = "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"
os.environ["TAVILY_API_KEY"] = tavily_api_key

search_tool = TavilySearch(
    max_results=5
)

api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=1000
)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
