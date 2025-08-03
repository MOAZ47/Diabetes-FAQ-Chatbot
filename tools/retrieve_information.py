#from langchain.tools import BaseTool
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, PrivateAttr

import time
from datetime import datetime
from utils.logger import init_logging

logger = init_logging("retrieve_info_tool.log")


MAX_RETRIES = 3
REQUEST_DELAY = 2.0  # seconds between API requests



class QueryInput(BaseModel):
    query: str = Field(..., description="User query about diabetes")

class RetrieveInformation(BaseTool):
    name: str = "diabetes_faq_retriever"
    description: str = "Useful for answering diabetes-related questions using vector db as source"
    # args_schema: Type[BaseModel] = QueryInput
    # _qa_chain: any = PrivateAttr()
    # _query_count: int = PrivateAttr(default=0)

    def __init__(self, qa_chain, **kwargs):
        super().__init__(**kwargs)
        self._qa_chain = qa_chain
        # self._query_count = 0
        # logger.info(f"Initialized {self.name} tool with QA chain")

    
    def _run(self, query: str) -> str:
        start_time = time.time()
        self._query_count += 1
        logger.info(f"Processing query #{self._query_count}: {query[:30]}...")
        
        try:
            result = self._qa_chain.invoke({"query": query})
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"Successfully processed query #{self._query_count} "
                f"in {processing_time:.2f}s. "
                f"Result length: {len(result)} characters"
            )
            return result
            
        except Exception as e:
            logger.error(
                f"Failed to process query #{self._query_count}: {str(e)}",
                exc_info=True
            )
            raise

    def _arun(self, query: str):
        logger.warning("Async operation requested but not implemented")
        raise NotImplementedError("Async not supported")

    def __del__(self):
        logger.info(
            f"RetrieveInformation tool shutdown. "
            f"Total queries processed: {self._query_count}"
        )
