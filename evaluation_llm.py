import json, os
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
load_dotenv()

# ------------------------------
# LLM Setup (GPT-4 recommended)
# ------------------------------



model_name = "mistral-small-2506"

llm = ChatMistralAI(model= model_name, api_key= os.getenv('MISTRAL_API_KEY'))

# -----------------------------------
# Load sample predictions
# -----------------------------------

SAMPLE_FILE = "logs/test_samples.json"

if not os.path.exists(SAMPLE_FILE):
    print(f"❌ Test sample file not found: {SAMPLE_FILE}")
    exit(1)

with open(SAMPLE_FILE, "r") as f:
    test_samples = json.load(f)

# ------------------------------
# Evaluation Prompt
# ------------------------------
def build_eval_prompt(sample):
    return f"""
You're a medical QA evaluator.

Evaluate the **assistant's answer** to the user query below.

---

**Query:** {sample['query']}

**Answer from assistant:** {sample['answer']}

**Reference (ground truth):** {sample['reference']}
**Source used by assistant:** {sample['context_source']}

---

Evaluate the following:

1. **Truthfulness** (Is it medically correct?)
2. **Hallucination** (Did the assistant make up facts?)
3. **Grounding** (Does it match the context or real data?)
4. **Clarity** (Is the response understandable and helpful?)

Respond in this JSON format:

{{
  "truthfulness": "Excellent / Good / Poor",
  "hallucination": "None / Minor / Major",
  "grounding": "Strong / Weak / Missing",
  "clarity": "Clear / Confusing / Rambling",
  "comment": "Optional feedback"
}}
"""

# ------------------------------
# Run Evaluation
# ------------------------------
print("🔍 Running LLM-based Evaluation...\n")
results = []

for sample in test_samples:
    prompt = build_eval_prompt(sample)
    msg = HumanMessage(content=prompt)

    eval_result = llm([msg]).content
    print(f"--- {sample['query']} ---\n{eval_result}\n")

    results.append({
        "query": sample["query"],
        "evaluation": eval_result
    })

# ------------------------------
# Save Results
# ------------------------------
with open("logs/eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Saved evaluation to logs/eval_results.json")
