from trulens.core import TruSession, Feedback
from trulens.providers.litellm import LiteLLM
from trulens.apps.app import TruApp
from langchain.schema import HumanMessage
from bot import get_diabetes_info
import os, json
from dotenv import load_dotenv

# Initialize
load_dotenv()
tru = TruSession()

# 1. Set up your chain exactly as shown in the quickstart
llm = LiteLLM(
    model="mistral-small-2506",
    api_key=os.getenv('MISTRAL_API_KEY'),
    temperature=0.2
)

# -----------------------------------
# Load sample predictions
# -----------------------------------

SAMPLE_FILE = "logs/test_samples.json"

if not os.path.exists(SAMPLE_FILE):
    print(f"❌ Test sample file not found: {SAMPLE_FILE}")
    exit(1)

with open(SAMPLE_FILE, "r") as f:
    test_samples = json.load(f)

# 2. Define your actual QA function
class DiabetesQAApp:
    def __init__(self):
        # Initialize any required components here
        pass
    
    def __call__(self, query: str) -> str:
        """This is where your actual QA logic should live"""
        # Replace this with your real implementation
        # For example:
        from bot import get_diabetes_info
        return get_diabetes_info(query)

# 3. Create instance of your app
app = DiabetesQAApp()


# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(
        llm.relevance_with_cot_reasons, name="Answer Relevance"
    ).on_input_output()
)



test_queries = [
        "what are symptoms of type 2 diabetes?",
        "for an indian person suffering with diabetes, suggest some breakfast"
    ]

tru_chain = TruApp(
    app,
    app_name="Eval-App",
    feedbacks=[f_answer_relevance]
)
for query in test_queries:
    with tru_chain as recording:
        response = app(query)

tru.get_leaderboard()








# Define the feedback function
f_answer_relevance = Feedback(
    llm_provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input_output()

golden_set = [
    {
        "query": "What are symptoms of type 2 diabetes?",
        "expected_response": "The symptoms include thirst, urination, blurry vision.",
    },
    {
        "query": "Can diabetics eat mango?",
        "expected_response": "Yes, they can eat it every day.",
    },
]

f_groundtruth = Feedback(
    GroundTruthAgreement(golden_set, provider=llm_provider).agreement_measure,
    name="Ground Truth Semantic Agreement",
).on_input_output()


# Define the app wrapper with a single callable function
class DiabetesApp:
    def __call__(self, query: str) -> str:
        return get_diabetes_info(query)

    @property
    def _call(self):  # ✅ TruBasicApp looks for this!
        return self.__call__

# Instantiate your wrapper
app_instance = DiabetesApp()

@tru_wrap
class TruDiabetesApp(DiabetesApp): # Naya naam de sakte ho ya same bhi rakh sakte ho
    pass

# Create TruBasicApp evaluator
tru_app = TruBasicApp(
    app=app_instance,
    app_id="diabetes-faq-app",
    feedbacks=[f_answer_relevance]
)

# Instrumented query engine can operate as a context manager:
with tru_app as recording:
    app_instance.completion("who invented the lightbulb?")

tru.get_leaderboard(app_ids=[tru_app.app_id])


# --------------------------------------
# Pinecone Vector DB Setup
# --------------------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), pool_threads=1)
index_name = "diabetes-faq"

# Delete old index
if index_name in pc.list_indexes():
    print(f"Deleting existing index: {index_name}")
    pc.delete_index(index_name)

# Create new index
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec={"cloud": "aws", "region": "us-east-1"}
    )

# Load embeddings
try:
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=os.getenv("MISTRAL_API_KEY"))
except:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# Load and split documents
diabetes_urls = [
    "https://www.jagran.com/health/diabetes",
    "https://www.diabetes.org/diabetes",
    "https://www.cdc.gov/diabetes/index.html",
    "https://www.niddk.nih.gov/health-information/diabetes"
]
loader = WebBaseLoader(
    diabetes_urls,
    requests_kwargs={"headers": {"User-Agent": "Mozilla/5.0"}}
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# Upload to Pinecone
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

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)