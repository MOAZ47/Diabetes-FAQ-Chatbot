
import os
import sys
import logging
import pytz
import time
from uuid import uuid4
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# --- Absolute Path Setup ---
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "vector_db.log")

#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
#LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Create logs directory if it doesn't exist
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except Exception as e:
    print(f"Failed to create log directory: {str(e)}")
    raise

#LOG_PATH = os.path.join(LOG_DIR, "vector_db.log")

# --- Enhanced Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info(f"Script started (PID: {os.getpid()})")
logger.info(f"Log file location: {os.path.abspath(LOG_PATH)}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_documents_with_retry(urls):
    """Load documents with retry logic and detailed logging"""
    try:
        start_time = time.time()
        loader = WebBaseLoader(urls, requests_kwargs={
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }
        },
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents with {sum(len(doc.page_content) for doc in documents)} total characters in {time.time()-start_time:.2f}s")
        return documents
    except Exception as e:
        logger.error(f"Failed to load documents: {str(e)}")
        raise

def get_embeddings():
    """Get embeddings with fallback and detailed logging"""
    try:
        logger.info("Attempting to initialize MistralAI embeddings")
        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            api_key=os.getenv("MISTRAL_API_KEY")
        )
        logger.info("Successfully initialized MistralAI embeddings")
        return embeddings
    except Exception as e:
        logger.warning(f"MistralAI embeddings failed: {str(e)}")
        logger.info("Falling back to HuggingFace embeddings (all-MiniLM-L6-v2)")
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

def create_db():
    """Main function to create Pinecone vector database"""
    total_start = time.time()
    logger.info("[START] Initializing Pinecone vector db creation")
    
    # List of authoritative diabetes websites
    diabetes_urls = [
        # 🇺🇸 US Government and Non-Profit Sources
        "https://www.diabetes.org/diabetes",                          # American Diabetes Association
        "https://www.cdc.gov/diabetes/index.html",                    # Centers for Disease Control and Prevention
        "https://www.niddk.nih.gov/health-information/diabetes",     # NIH - NIDDK

        # 🇮🇳 Indian Sources
        "https://www.jagran.com/health/diabetes",                     # Jagran (Hindi, popular Indian health portal)

        # 🌍 International Organizations
        "https://www.who.int/health-topics/diabetes",                 # World Health Organization (WHO)
        "https://www.nhs.uk/conditions/diabetes/",                    # NHS UK – Diabetes overview

        # 🧪 Research and Education Portals
        "https://www.mayoclinic.org/diseases-conditions/diabetes",   # Mayo Clinic

        # 📲 Public-friendly Indian resources
        "https://www.lybrate.com/topic/diabetes",                    # Lybrate (popular in India for telehealth)
        "https://www.jagran.com/health/diabetes",
        "https://www.diabetes.org/diabetes",
        "https://www.cdc.gov/diabetes/index.html",
        "https://www.niddk.nih.gov/health-information/diabetes",

        'https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444',
        'https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/symptoms-causes/syc-20351193',
        'https://www.mayoclinic.org/diseases-conditions/gestational-diabetes/symptoms-causes/syc-20355339',
        'https://www.mayoclinic.org/diseases-conditions/type-1-diabetes-in-children/symptoms-causes/syc-20355306',
        'https://www.mayoclinic.org/diseases-conditions/type-1-diabetes/symptoms-causes/syc-20353011',
        'https://www.mayoclinic.org/diseases-conditions/type-2-diabetes-in-children/symptoms-causes/syc-20355318',
        'https://www.mayoclinic.org/diseases-conditions/hyperglycemia/symptoms-causes/syc-20373631',
        'https://www.mayoclinic.org/diseases-conditions/diabetes-insipidus/symptoms-causes/syc-20351269',
        'https://www.mayoclinic.org/diseases-conditions/diabetic-nephropathy/symptoms-causes/syc-20354556',
        'https://www.mayoclinic.org/diseases-conditions/diabetic-retinopathy/symptoms-causes/syc-20371611',
        
    ]

    try:
        pc_start = time.time()
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), pool_threads=10)
        index_name = "diabetes-faq"
        logger.info(f"Pinecone initiated in {time.time() - pc_start:.2f}s")

        # Delete old index
        if index_name in pc.list_indexes():
            print(f"Deleting existing index: {index_name}")
            pc.delete_index(index_name)
        
        # Create new index
        index_start = time.time()
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec={"cloud": "aws", "region": "us-east-1"}
            )
        logger.info(f"Created Pinecone index in {time.time() - index_start:.2f}s")

        # Load documents
        documents = load_documents_with_retry(diabetes_urls)
        
        # Split documents
        split_start = time.time()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks (avg: {sum(len(text.page_content) for text in chunks)/len(chunks):.0f} chars) in {time.time()-split_start:.2f}s")

        # Get embeddings
        embed_start = time.time()
        embeddings = get_embeddings()
        logger.info(f"Embeddings initialized in {time.time()-embed_start:.2f}s")

        # Upload to Pinecone
        upload_start = time.time()
        records = []
        for chunk in chunks:
            record = {
                "_id": str(uuid4()),
                "chunk_text": chunk.page_content,
                **chunk.metadata
            }
            records.append(record)

        index = pc.Index(index_name)
        index.upsert_records(namespace="ns1", records=records)
        logger.info(f"Uploaded {len(chunks)} vectors in {time.time() - upload_start:.2f}s")
        
        
        # Final logging
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist).strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
        total_time = time.time() - total_start
        logger.info(f"[DONE] Completed in {total_time:.2f} seconds at {now}")
        logger.info(f"Pinecone index: {index_name}")

    except Exception as e:
        logger.error(f"Critical error in create_db: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Script execution completed")

if __name__ == "__main__":
    create_db()