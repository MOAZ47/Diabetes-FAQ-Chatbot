
import os
import logging
import pytz
import time
from uuid import uuid4
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.schema import Document

from pinecone import Pinecone, IndexEmbed, EmbedModel, Metric
from typing import List, Tuple
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


def load_documents_with_retry(urls: List[str]) -> Tuple[List[Document], List[str]]:
    """
    Load documents with retry logic that skips failed URLs after max attempts.
    Returns a tuple of (list of successfully loaded Document objects, list of failed URLs).
    Each Document object is properly formatted for LangChain text splitters.
    """
    successful_docs = []
    failed_urls = []
    
    for url in urls:
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            reraise=False  # Don't re-raise after last attempt
        )
        def load_single_url(current_url: str) -> List[Document]:
            try:
                start_time = time.time()
                loader = WebBaseLoader(
                    [current_url],  # Wrapping single URL in list
                    requests_kwargs={
                        "headers": {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
                        }
                    }
                )
                docs = loader.load()
                
                # Validate document structure
                if not docs or not isinstance(docs[0], Document):
                    raise ValueError(f"Invalid document format from {current_url}")
                
                logger.info(f"Successfully loaded {current_url} with {len(docs[0].page_content)} chars in {time.time()-start_time:.2f}s")
                return docs
            except Exception as e:
                logger.warning(f"Attempt failed for {current_url}: {str(e)}")
                raise

        try:
            docs = load_single_url(url)
            successful_docs.extend(docs)
        except "RetryError":
            logger.error(f"Max retries (3) reached for URL: {url}. Skipping.")
            failed_urls.append(url)
        except Exception as e:
            logger.error(f"Unexpected error loading {url}: {str(e)}. Skipping.")
            failed_urls.append(url)
    
    # Final validation before return
    if not all(isinstance(doc, Document) for doc in successful_docs):
        raise ValueError("Some loaded documents are not valid Document objects")
    
    if failed_urls:
        logger.warning(f"Failed to load {len(failed_urls)} URLs: {failed_urls}")
    
    return successful_docs, failed_urls


# def get_embeddings():
#     """Get embeddings with fallback and detailed logging"""
#     try:
#         logger.info("Attempting to initialize MistralAI embeddings")
#         embeddings = MistralAIEmbeddings(
#             model="mistral-embed",
#             api_key=os.getenv("MISTRAL_API_KEY")
#         )
#         logger.info("Successfully initialized MistralAI embeddings")
#         return embeddings
#     except Exception as e:
#         logger.warning(f"MistralAI embeddings failed: {str(e)}")
#         logger.info("Falling back to HuggingFace embeddings (all-MiniLM-L6-v2)")
#         return HuggingFaceEmbeddings(
#             model_name="all-MiniLM-L6-v2",
#             model_kwargs={'device': 'cpu'},
#             encode_kwargs={'normalize_embeddings': True}
#         )

def batch_upsert_records(index, records: list, namespace: str, batch_size: int = 96):
    """Upsert records in batches to comply with Pinecone's limits"""
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            index.upsert_records(records=batch, namespace=namespace)
            logger.info(f"Successfully upserted batch {i//batch_size + 1} with {len(batch)} records")
        except Exception as e:
            logger.error(f"Failed to upsert batch {i//batch_size + 1}: {str(e)}")
            raise


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

        # 📲 Public-friendly Indian resources
        "https://www.lybrate.com/topic/diabetes",                    # Lybrate (popular in India for telehealth)

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
        pc = Pinecone(api_key= os.getenv("PINECONE_API_KEY"), pool_threads=4)
        index_name = "diabetes-faq"
        logger.info(f"Pinecone initiated in {time.time() - pc_start:.2f}s")
        
        # Create new index
        index_start = time.time()
        if not pc.has_index(index_name):
            logger.info(f"Pinecone index NOT FOUND, creating new index.")
            pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed= IndexEmbed(
                    model=EmbedModel.Multilingual_E5_Large,
                    metric=Metric.COSINE,
                    field_map={"text": "chunk_text"}  # 👈 this tells Pinecone: "Embed metadata['description']"
                )
            )
            logger.info(f"Created Pinecone index in {time.time() - index_start:.2f}s. \
                        Built index with following info: \n{pc.describe_index(index_name)}")
            logger.info(f"\nUsing PINECONE's inbuilt embedding model")
        else:
            logger.info(f"Using existing Pinecone index: {index_name}, with inbuilt embedding model.")
            logger.info(f"Pinecone index description:\n{pc.describe_index(index_name)}")

        # Load documents
        documents, failed_urls = load_documents_with_retry(diabetes_urls)
        
        # Split documents
        split_start = time.time()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks (avg: {sum(len(text.page_content) for text in chunks)/len(chunks):.0f} chars) in {time.time()-split_start:.2f}s")

        # Create Records expected by Pinecone
        records = []
        for chunk in chunks:
            records.append({
                "id": str(uuid4()),
                "chunk_text": chunk.page_content,  # Field that will be embedded
                
            })
        logger.info(f"Created {len(records)} records for uploading to Pinecone")

        # Upload to Pinecone
        upload_start = time.time()
        logger.info("Starting upload to PineCone")

        index_host = pc.describe_index(index_name).host

        index = pc.Index(host= index_host)

        # index.upsert_records(records=records, namespace="ns1")
        # Upsert in batches instead of all at once
        batch_upsert_records(index, records, "ns1")

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