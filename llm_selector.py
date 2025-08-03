import os
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
load_dotenv()

# Model options with priority order
MODEL_OPTIONS = {
    "mistral": [
        "mistral-medium-latest",
        "mistral-small-latest",
        "mistral-large-latest"
    ],
    "cohere": ["command-r", "command-r-plus"]
}

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
def test_model_works(llm):
    """Test if the model can actually perform inference"""
    test_messages = [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="Translate: I love programming")
    ]
    try:
        # Real inference test
        response = llm.invoke(test_messages)
        # Check if we got a valid response
        if not response or not response.content:
            raise ValueError("Empty response")
        # Simple content check
        if "programmation" not in response.content.lower():  # Expected French translation
            raise ValueError("Unexpected response content")
        return True
    except Exception as e:
        print(f"Model test failed: {str(e)}")
        return False

def get_working_llm():
    """Return the first working LLM instance that passes actual inference test"""
    # Try Mistral models first
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if mistral_api_key:
        for model in MODEL_OPTIONS["mistral"]:
            try:
                llm = ChatMistralAI(
                    model=model,
                    api_key=mistral_api_key,
                    temperature=0.7
                )
                if test_model_works(llm):
                    print(f"✅ Verified working Mistral model: {model}")
                    return llm
            except Exception as e:
                print(f"⚠️ Mistral model {model} failed: {str(e)}")
                continue
    
    # Fallback to Cohere if Mistral fails
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if cohere_api_key:
        for model in MODEL_OPTIONS["cohere"]:
            try:
                llm = ChatCohere(
                    model=model,
                    temperature=0.2,
                    cohere_api_key=cohere_api_key
                )
                if test_model_works(llm):
                    print(f"✅ Verified working Cohere model: {model}")
                    return llm
            except Exception as e:
                print(f"⚠️ Cohere model {model} failed: {str(e)}")
                continue
    
    raise ValueError("❌ No working LLM found - check API keys and models")



if __name__ == "__main__":
    # Initialize LLM
    llm_model = get_working_llm()

    messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="I love programming.")
    ]
    ai_msg = llm_model.invoke(messages)
    print("Translation:", ai_msg.content)