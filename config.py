import os
from crewai import LLM
from langchain_mistralai import MistralAIEmbeddings

MODEL_NAME = "mistral/mistral-medium-latest"
INDEX_NAME = "diabetes-faq"

llm_model = LLM(
    model=MODEL_NAME,
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.3
)

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.getenv("MISTRAL_API_KEY")
)
