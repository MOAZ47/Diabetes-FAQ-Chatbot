
import os, time
from datetime import datetime
from dotenv import load_dotenv

# Langchain/CrewAI imports
from crewai import Agent, Task, Crew, LLM
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore

# Huggingface import
from langchain_community.embeddings import HuggingFaceEmbeddings

# Local imports
from tools.retrieve_information import RetrieveInformation
from utils.logger import init_logging

# Initialize environment
load_dotenv()

logger = init_logging('medical_research.log')
    

# --- Setup ---
MODEL_NAME = "mistral/mistral-medium-latest"
# VECTORSTORE_PATH = "vectorstore"
INDEX_NAME = "diabetes-faq"
# MAX_RETRIES = 3
REQUEST_DELAY = 2.0  # seconds between API requests


def initialize_components():
    """
        Initialize all LLM, Pinecone vectorstore and retriever.
    
    """
    try:
        start_time = time.time()
        
        # Initialize LLM with API key validation
        if not os.getenv("MISTRAL_API_KEY"):
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
            
        logger.info("Initializing Crew ai LLM model")
        llm_model = LLM(model = MODEL_NAME, 
                        api_key = os.getenv("MISTRAL_API_KEY"),
                        temperature = 0.3)
        logger.info(f"Crew ai LLM model initialized in {time.time()-start_time:.2f}s")
        
        # Load Vector Store with fallback
        
        load_start = time.time()
        
        logger.info("Loading Embeddings ...")
        try:
            
            embeddings = MistralAIEmbeddings(
                model="mistral-embed",
                api_key=os.getenv("MISTRAL_API_KEY")
            )
            logger.info("Succesfully loaded Mistral Embeddings")
            
        except Exception as e:
            logger.warning(f"MistralAI embeddings failed: {str(e)}")
            logger.info("Falling back to HuggingFace embeddings")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        logger.info("Loading Pinecone vectors ...")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name = INDEX_NAME,
            embedding = embeddings,
            namespace = "ns1",
            text_key = "chunk_text"
        )
        
        logger.info(f"Loaded Pinecone Vector store in {time.time()-load_start:.2f}s")
        
        # Initialize Chat Model with delay
        time.sleep(REQUEST_DELAY)
        logger.info("Creating QA chain ...")

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
        
        rt = RetrieveInformation(qa_chain = qa_chain)
        
        return llm_model, vectorstore, rt
        
    except Exception as e:
        logger.error(f"Component initialization failed: {str(e)}")
        raise

def create_medical_researcher():
    try:
        logger.info("Starting medical researcher agent ...")
        setup_start = time.time()
        
        # Initialize all components with retry logic
        llm_model, vectorstore, retrieval_tool = initialize_components()
        
        # Create Agent
        medical_researcher = Agent(
            role="Senior Diabetes Researcher",
            goal="Provide accurate, well-sourced medical information about diabetes",
            backstory=(
                "Board-certified endocrinologist with 15 years of clinical experience "
                "at AIIMS Delhi. Specializes in evidence-based diabetes management."
            ),
            tools = [retrieval_tool],
            llm = llm_model,
            memory = True,
            max_iter = 10,
            verbose = False, 
        )
        
        logger.info(f"Medical researcher agent setup completed in {time.time()-setup_start:.2f}s")

        return medical_researcher
        
    except Exception as e:
        logger.critical(f"Failed to initialize medical researcher: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info(f"Script execution completed at {datetime.now().isoformat()}")


# --- Test Block ---
if __name__ == "__main__":
    try:
        
        logger.info("Creating medical researcher agent...")
        researcher = create_medical_researcher()
        
        # Test the agent
        test_query = "What are the early symptoms of type 2 diabetes?"
        logger.info(f"Testing with query: {test_query}")
        
        task = Task(
            description=f"Research: {test_query}",
            agent=researcher,
            expected_output="Concise summary of early symptoms"
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
__all__ = ['medical_researcher']