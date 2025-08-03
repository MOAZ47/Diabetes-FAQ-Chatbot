from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, PrivateAttr
from tavily import TavilyClient
import os
from dotenv import load_dotenv

import time
from datetime import datetime
from utils.logger import init_logging

load_dotenv()

logger = init_logging("web_search_tool.log")


# Configuration
MAX_RETRIES = 3
REQUEST_DELAY = 2.0  # seconds between API requests
SEARCH_TIMEOUT = 10  # seconds



class WebQueryInput(BaseModel):
    query: str = Field(..., description="User's health-related search query")

class WebSearchTool(BaseTool):
    name: str = "health_web_search"
    description: str = "Helpful for getting real-time health-related info from the web"
    args_schema: Type[BaseModel] = WebQueryInput
    _client: TavilyClient = PrivateAttr()
    _query_count: int = PrivateAttr(default=0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            start_time = time.time()
            
            self._client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            self._query_count = 0

            logger.info(f"WebSearchTool initialized in {time.time()-start_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to initialize WebSearchTool: {str(e)}", exc_info=True)
            raise
    
    def _run(self, query: str) -> str:
        try:
            start_time = time.time()
            self._query_count += 1
            current_query = self._query_count
            
            logger.info(f"[Query #{current_query}] Processing: {query[:100]}...")
            logger.debug(f"Full query: {query}")
            
            # Add delay between requests
            time.sleep(REQUEST_DELAY)
            
            # Execute search
            search_start = time.time()
            result = self._client.search(
                query=query,
                include_answer=True,
                max_results=3,
                # search_depth="basic",
                # timeout=SEARCH_TIMEOUT
            )
            
            processing_time = time.time() - start_time
            search_time = time.time() - search_start
            
            # Analyze and log results
            answer = result.get("answer", "No answer found.")
            tuple_selector = ([result['results'][_]['url'] for _ in range(len(result['results']))], [])
            sources = tuple_selector[0] if tuple_selector[0] else tuple_selector[1]
            
            logger.info(
                f"[Query #{current_query}] Completed in {processing_time:.2f}s "
                f"(Search: {search_time:.2f}s). "
                f"Answer length: {len(answer)} chars. "
                f"Sources found: {len(sources)}"
            )
            logger.debug(f"Answer preview: {answer[:200]}...")
            
            return answer
            
        except Exception as e:
            logger.error(
                f"[Query #{current_query}] Failed after {time.time()-start_time:.2f}s: {str(e)}",
                exc_info=True
            )
            raise

    def _arun(self, query: str):
        logger.warning("Async operation requested but not implemented")
        raise NotImplementedError("Async not supported")

    def __del__(self):
        logger.info(
            f"WebSearchTool shutdown. Total queries processed: {self._query_count}"
        )