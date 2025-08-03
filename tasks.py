from crewai import Task
from agents.medical_researcher import create_medical_researcher #medical_researcher
from agents.health_coach import create_web_search_tool #health_coach


from utils.logger import init_logging

logger = init_logging("tasks.log")

medical_researcher = create_medical_researcher()

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
        "4. Use proper markdown including bullet points and tables.\n"
        "5. If there is any word or character limitation in query then follow it, else keep the answer to a maximum of 200 words."
    ),
    agent = medical_researcher, 
    output_file = "logs/medical_research.txt"
)

# ---------------------------------------------------------------------
health_coach = create_web_search_tool()

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
        "5. Use proper markdown including bullet points and tables.\n"
        "5. If there is any word or character limitation in query then follow it, else keep the answer to a maximum of 200 words."
    ),
    agent = health_coach,
    output_file = "logs/lifestyle_advice.txt"
)