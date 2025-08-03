
import os
import time
from datetime import datetime
from dotenv import load_dotenv
# from tenacity import retry, stop_after_attempt, wait_exponential

# Langchain/CrewAI imports
from crewai import Agent, Task, Crew, LLM
# from langchain_mistralai.chat_models import ChatMistralAI
# from langchain_cohere import ChatCohere

# Local imports
from tools.web_search_tool import WebSearchTool
from utils.logger import init_logging

load_dotenv()

logger = init_logging("health_coach.log")


# --- Configuration ---
MODEL_NAME = "mistral/mistral-medium-latest"
REQUEST_DELAY = 2.0  # seconds between API requests


def initialize_llm():
    """
        Initialize LLM.
    
    """
    try:
        start_time = time.time()
        
        # Validate API key
        if not os.getenv("MISTRAL_API_KEY"):
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
            
        logger.info("Initializing Crew ai LLM model")
        llm_model = LLM(model = MODEL_NAME, 
                        api_key = os.getenv("MISTRAL_API_KEY"),
                        temperature = 0.3)
        
        logger.info(f"Crew ai Health Coach LLM model initialized in {time.time()-start_time:.2f}s")
        return llm_model
        
    except Exception as e:
        logger.error(f"LLM initialization failed: {str(e)}")
        raise


def initialize_web_tool():
    """
        Initialize web search tool with retry logic
    """
    try:
        logger.info("Initializing WebSearchTool")
        start_time = time.time()
        tool = WebSearchTool()
        logger.info(f"WebSearchTool initialized in {time.time()-start_time:.2f}s")
        return tool
    except Exception as e:
        logger.error(f"WebSearchTool initialization failed: {str(e)}")
        raise

def create_web_search_tool():
    try:
        total_start = time.time()
        logger.info("Starting Health Coach agent setup")
        
        # Initialize components
        llm_model = initialize_llm()
        web_tool = initialize_web_tool()
        
        # Create Agent
        agent_start = time.time()
        logger.info("Creating Health Coach agent")
        
        health_coach = Agent(
            role= "Diabetes Lifestyle Coach",
            goal= (
                "Provide practical, culturally-appropriate lifestyle advice "
                "for Indian diabetics with portion sizes and meal timing"
            ),
            backstory=(
                "Certified diabetes educator and nutritionist with 10 years experience "
                "helping Indian patients manage diabetes through diet and exercise."
            ),
            tools=[web_tool],
            llm=llm_model, 
            verbose=False
        )
        
        logger.info(f"Health Coach agent created in {time.time()-agent_start:.2f}s")
        logger.info(f"Total setup time: {time.time()-total_start:.2f}s")
        logger.info("Health Coach agent successfully initialized")

        return health_coach

    except Exception as e:
        logger.critical(f"Failed to initialize Health Coach agent: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info(f"Script execution completed at {datetime.now().isoformat()}")


if __name__ == "__main__":
    try:
        
        logger.info("Creating medical researcher agent ... ")
        researcher = create_web_search_tool()
        
        # Test the agent
        test_query = "What food items to avoid if suffering from Diabetes?"
        logger.info(f"Testing with query: {test_query}")
        
        task = Task(
            description = f"Provide lifestyle recommendations for: {test_query}",
            agent = researcher,
            expected_output = "Practical advice about food items, cooking tips, etc"
        )
        
        crew = Crew(agents=[researcher], tasks=[task])
        result = crew.kickoff()
        
        logger.info("\n--- Result ---")
        logger.info(result)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
    finally:
        logger.info("Execution completed")

# Make agent available for import
__all__ = ['health_coach']