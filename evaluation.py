import os
from dotenv import load_dotenv
import numpy as np

from bot import get_diabetes_info, qa_chain
#from main import get_diabetes_info

# TruLens setup
from trulens.core import TruSession, Feedback, FeedbackMode, Select
#from trulens.dashboard.run import run_dashboard
#from trulens.dashboard import run_dashboard
from trulens.providers.litellm import LiteLLM
from trulens.apps.app import TruApp, instrument
from trulens.core.guardrails.base import block_output

# import weave

# from utils.logger import setup_logger

# eval_logger = setup_logger("eval", "trulens_eval.log")
# weave_logger = setup_logger("weave", "weave_monitor.log")

# Load env
load_dotenv()

# weave.init("diabetes-qa-evaluation")

# --------------------------------------
# Init TruLens session
# --------------------------------------
tru = TruSession()
tru.reset_database()

TEMP = 0.1
llm_provider = LiteLLM(
    model_engine="mistral/mistral-small-2506",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature = TEMP
)



# --------------------------------------
# Custom App with Instrumentation
# --------------------------------------

_feedback = Feedback(llm_provider.criminality, higher_is_better = False)

class DiabetesApp:
    @instrument
    # @block_output(feedback = _feedback,
    #     threshold = 0.5,
    #     return_value = "Sorry, I couldn't find an answer to your question.")
    def call(self, query: str) -> str:
        return get_diabetes_info(query)

llm_app = DiabetesApp()

# --------------------------------------
# Feedback Functions
# --------------------------------------

f_harm = (Feedback(llm_provider.harmfulness, name="Hate Speech Detection", higher_is_better=False)
          .on_output())
f_malicious = (Feedback(llm_provider.maliciousness_with_cot_reasons, name="Malicious", higher_is_better=False).on_output())
f_criminal = (Feedback(llm_provider.criminality_with_cot_reasons, name="Criminal", higher_is_better=False).on_output())

f_answer_relevance = (
    Feedback(llm_provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on(Select.RecordCalls)  # User query
    .on(Select.RecordOutput)   # Model response
    .aggregate(np.mean)      # Aggregation method
)



f_groundedness = (
    Feedback(
        llm_provider.groundedness_measure_with_cot_reasons, name="Groundedness - LLM Judge"
    )
    .on(Select.RecordCalls)
    .on(Select.RecordOutput)
)

f_context_relevance = (
    Feedback(llm_provider.context_relevance_with_cot_reasons, name="Context Relevance")
    .on(Select.RecordCalls)
    .on(Select.RecordOutput)  
    .aggregate(np.mean)
)

f_coherence = (
    Feedback(llm_provider.coherence_with_cot_reasons, name="Coherence")
    .on(Select.RecordOutput) 
)

f_comprehensiveness = (
    Feedback(
        llm_provider.comprehensiveness_with_cot_reasons, name="Comprehensiveness"
    )
    .on_input()
    .on_output()
)


# --------------------------------------
# Register app with TruLens
# --------------------------------------
APP_NAME = "diabetes-auto-app"
APP_VERSION = "v1"
APP_ID = "diabetes-auto-app-v1"

_feedbacks = [
        f_answer_relevance,
        f_groundedness,
        f_context_relevance,
        f_coherence,
    ]

tru_app = TruApp(
    app=llm_app,
    app_id=APP_ID,
    app_name=APP_NAME,
    app_version=APP_VERSION,
    feedbacks= _feedbacks,
    #feedback_mode=FeedbackMode.WITH_APP_THREAD
)

# --------------------------------------
# Run Evaluation
# --------------------------------------
# 🧪 Run test queries
test_queries = [
    "What is type 1 diabetes?",
    "Suggest Indian breakfast for diabetics"

    # "What is type 1 diabetes?",
    # "Suggest Indian breakfast for diabetics",
    #"What are the early symptoms of diabetes?",
    # "Can diabetes be reversed with lifestyle changes?",
    #"How does insulin resistance work in type 2 diabetes?",
    # "What is the ideal blood sugar level after meals?",
    # "Which fruits should diabetics avoid?"
]

#tru.add_app(app=tru_app)

def run_evaluation():
    for query in test_queries:
        print("---------------------------------")
        print(f"\nEvaluating: '{query}'")

        _response, record = tru_app.with_record(llm_app.call, query)

        tru.add_app(app=tru_app)
        tru.add_record(record)

        feedback_results = tru.run_feedback_functions(
            record=record,
            app=tru_app,
            feedback_functions= _feedbacks
        )
        tru_app.wait_for_feedback_results(feedback_timeout=60)
        tru.add_feedbacks(feedback_results)

        print(f"Is thread active: {tru_app.manage_pending_feedback_results_thread.is_alive()}")

        #tru_app.db.update()

    print("\nResults:")
    _app_id = tru_app.app_id
    print(f"\nThe App ID is: {tru_app.app_id} and its type is {type(tru_app.app_id)}")
    #print(tru.get_leaderboard(app_ids=[_app_id]))

    print("\nSaving Leaderboard to json")
    tru.get_leaderboard(app_ids=[tru_app.app_id]).to_json(path_or_buf="./logs/trulens_eval_results.json", indent=4)

    records_df, feedback_df = tru.get_records_and_feedback(app_ids=[tru_app.app_id])

    print("\nTotal no. of records and feedbacks")
    print(f"Records: {len(records_df)}")
    print(f"Feedback: {len(feedback_df)}")

    # print("\nResults")
    # print(records_df.iloc[0])
    # Optional
    # tru.run_dashboard()


# Only run if this file is executed directly (not on import)
if __name__ == "__main__":
    run_evaluation()
