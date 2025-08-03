import os
import logging
from typing import Type
from pydantic import BaseModel, Field, PrivateAttr
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import BaseTool
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from tavily import TavilyClient

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "bot.log")

# Simple logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_PATH, mode='a'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "mistral/mistral-medium-latest"
VECTORSTORE_PATH = "vectorstore"
INDEX_NAME = "diabetes-faq"

try:
    # Initialize LLM and embeddings
    logger.info("Initializing LLM and embeddings...")
    llm_model = LLM(model = MODEL_NAME, 
                    api_key = os.getenv("MISTRAL_API_KEY"),
                    temperature = 0.3)
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=os.getenv("MISTRAL_API_KEY"))

    # Load vector store
    logger.info(f"Loading Pinecone vectorstore: {INDEX_NAME} ...")

    # vectorstore = FAISS.load_local(
    #     VECTORSTORE_PATH, 
    #     embeddings, 
    #     index_name=INDEX_NAME,
    #     allow_dangerous_deserialization=True
    # )
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name = INDEX_NAME,
        embedding = embeddings,
        namespace = "ns1",
        text_key = "chunk_text"
    )

    # Create QA chain
    logger.info("Creating QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatMistralAI(model=MODEL_NAME, 
                          api_key=os.getenv("MISTRAL_API_KEY"),
                          temperature=0.3),
        retriever = vectorstore.as_retriever(
            search_type = 'mmr',
            search_kwargs={"k": 7, "score_threshold": 0.75, 'lambda_mult': 0.25}
        ),
        chain_type="stuff",
        return_source_documents=True
    )

except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    raise

#----------------------------------------------------
# CUSTOM TOOLS
#----------------------------------------------------
class RetrieveInformation(BaseTool):
    name: str = "diabetes_faq_retriever"
    description: str = "Useful for answering diabetes-related questions using vector db as source"

    def __init__(self, qa_chain, **kwargs):
        super().__init__(**kwargs)
        self._qa_chain = qa_chain

    def _run(self, query: str) -> str:
        logger.info(f"Processing query: {query[:50]}...")  # Log first 50 chars
        try:
            result = self._qa_chain.invoke({"query": query})

            docs = self._qa_chain.retriever.get_relevant_documents(query)  # Extract context

            logger.info("Query processed successfully")
            return {
                "answer": result,
                "context": [doc.page_content for doc in docs]  # For TruLens evaluation
            }
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return f"Error processing query: {str(e)}"

class WebQueryInput(BaseModel):
    query: str = Field(..., description="User's health-related search query")

class WebSearchTool(BaseTool):
    name: str = "health_web_search"
    description: str = "Helpful for getting real-time health-related info from the web"
    args_schema: Type[BaseModel] = WebQueryInput
    _client: TavilyClient = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def _run(self, query: str) -> str:
        logger.info(f"Web search for: {query[:50]}...")
        try:
            result = self._client.search(query=query, include_answer=True, max_results=3, )
            logger.info("Web search completed")
            return {
                "answer": result.get("answer", "No answer found."),
                "context": [result['results'][_]['content'] for _ in range(len(result['results']))]  # Web search context
            }
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return f"Search failed: {str(e)}"

# Initialize tools
logger.info("Initializing tools...")
retrieval_tool = RetrieveInformation(qa_chain=qa_chain)
web_tool = WebSearchTool()

#----------------------------------------------------
# A G E N T S
#----------------------------------------------------
logger.info("Setting up agents...")

medical_researcher = Agent(
    role="Senior Diabetes Researcher",
    goal="Provide accurate, well-sourced medical information about diabetes",
    backstory=(
        "Board-certified endocrinologist with 15 years of clinical experience "
        "at AIIMS Delhi. Specializes in evidence-based diabetes management."
    ),
    tools=[retrieval_tool],
    llm=llm_model,
    memory=True,
    max_iter=10,
    verbose=False, 
)

health_coach = Agent(
    role="Diabetes Lifestyle Coach",
    goal=(
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

#----------------------------------------------------
# T A S K S
#----------------------------------------------------
logger.info("Creating tasks...")

research_task = Task(
    description="""Research accurate information about: {query}
        - Include diagnostic criteria when relevant
        - Cite treatment guidelines from source
        - Note potential complications""",
    expected_output=(
        "Structured response with:\n"
        "1. Key medical facts\n"
        "2. Supporting sources (document excerpts)\n"
        "3. Severity indicators (if applicable)\n"
        "4. Use Bullet-point for proper structure."
    ),
    agent=medical_researcher, 
    output_file = "logs/medical_research.txt"
)

advice_task = Task(
    description="""Provide lifestyle recommendations for: {query}
        - Specify portion sizes in grams/cups
        - Include preparation methods
        - Suggest meal timing""",
    expected_output=(
        "Practical advice with:\n"
        "1. Food items with quantities\n"
        "2. Cooking tips\n"
        "3. Weekly frequency\n"
        "4. Web sources (if used)\n"
        "5. Use Bullet-point for proper structure."
    ),
    agent = health_coach,
    output_file = "logs/lifestyle_advice.txt"
)

#----------------------------------------------------
# C R E W
#----------------------------------------------------
logger.info("Assembling crew...")
diabetes_crew = Crew(
    agents=[medical_researcher, health_coach],
    tasks=[research_task, advice_task],
    process= Process.sequential,
    verbose=False
)

def get_diabetes_info(query):
    logger.info(f"[START] Starting Processing ...")
    logger.info(f"Processing user query: {query}")

    try:
        # Run the crew and capture full response
        result = diabetes_crew.kickoff(inputs={'query': query})
        logger.info("Crew execution completed")

        # Log the final combined result
        #print("\n🔚 Final Combined Output:\n", result)
        logger.info(f"Final Combined Output:\n{result}")

        # Check and log each agent’s output file (if you're writing to txt files)
        med_path = os.path.join("logs", "medical_research.txt")
        life_path = os.path.join("logs", "lifestyle_advice.txt")

        if os.path.exists(med_path):
            with open(med_path, "r", encoding="utf-8") as f:
                med_output = f.read()
                #print("\n🧬 Medical Researcher Output:\n", med_output)
                logger.info("Medical Researcher Output:\n" + med_output)
        else:
            #print("❌ No medical researcher output file found.")
            logger.warning("Missing medical researcher output file.")

        if os.path.exists(life_path):
            with open(life_path, "r", encoding="utf-8") as f:
                lifestyle_output = f.read()
                #print("\n🥗 Health Coach Output:\n", lifestyle_output)
                logger.info("Health Coach Output:\n" + lifestyle_output)
        else:
            print("❌ No health coach output file found.")
            logger.warning("Missing health coach output file.")

        return result

    except Exception as e:
        logger.error(f"System error while processing query: {str(e)}")
        return f"System error: {str(e)}"

if __name__ == "__main__":
    print("Diabetes Expert System (Mistral + CrewAI)")
    print("Type 'quit' to exit\n")
    
    test_query = "for an indian person suffering with diabetes, suggest some breakfast"
    logger.info("Starting test query...")
    response = get_diabetes_info(test_query)
    print(f"\nResponse:\n{response}\n")
    logger.info("Test completed")