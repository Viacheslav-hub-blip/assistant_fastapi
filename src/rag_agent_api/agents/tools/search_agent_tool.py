import os

from langchain_tavily import TavilySearch

tavily_api_key = "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"
os.environ["TAVILY_API_KEY"] = tavily_api_key

search_tool = TavilySearch(
    max_results=5
)
