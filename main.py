import os
from agents.medical_researcher import create_medical_researcher #medical_researcher
from agents.health_coach import create_web_search_tool #health_coach
from tasks import research_task, advice_task
from crewai import Crew, Process
from utils.logger import init_logging

logger = init_logging("main.log")

medical_researcher = create_medical_researcher()
health_coach = create_web_search_tool()

logger.info("Assembling crew...")

diabetes_crew = Crew(
    agents=[medical_researcher, health_coach],
    tasks=[research_task, advice_task],
    process= Process.sequential,
    verbose= False
)

def log_med_results(med_path):
    if os.path.exists(med_path):
        with open(med_path, "r", encoding="utf-8") as f:
            med_output = f.read()
            #print("\n🧬 Medical Researcher Output:\n", med_output)
            logger.info("Medical Researcher Output:\n" + med_output)
    else:
        #print("❌ No medical researcher output file found.")
        logger.warning("Missing medical researcher output file.")

def log_health_results(life_path):
    if os.path.exists(life_path):
        with open(life_path, "r", encoding="utf-8") as f:
            lifestyle_output = f.read()
            #print("\n🥗 Health Coach Output:\n", lifestyle_output)
            logger.info("Health Coach Output:\n" + lifestyle_output)
    else:
        print("❌ No health coach output file found.")
        logger.warning("Missing health coach output file.")

def classify_query(query: str) -> str:
    """
    Classifies user query into one of the following:
    - 'medical' for medical info
    - 'lifestyle' for lifestyle/diet advice
    - 'both' if both agents needed
    """
    query_lower = query.lower()

    medical_keywords = [
        "what is", "symptom", "diagnosis", "type 1", "type 2",
        "insulin", "treatment", "blood sugar", "glucose", "complication", "test"
    ]

    lifestyle_keywords = [
        "diet", "exercise", "lifestyle", "meal", "breakfast", "lunch", "dinner",
        "eat", "avoid", "nutrition", "food", "routine"
    ]

    medical = any(kw in query_lower for kw in medical_keywords)
    lifestyle = any(kw in query_lower for kw in lifestyle_keywords)

    if medical and lifestyle:
        return "both"
    elif medical:
        return "medical"
    elif lifestyle:
        return "lifestyle"
    else:
        return "both"

def get_diabetes_info(query):
    logger.info(f"[START] Starting Processing ...")

    category = classify_query(query)
    logger.info(f"Query classified as: {category}")
    logger.info(f"Processing user query: {query}")

    med_path = os.path.join("logs", "medical_research.txt")
    life_path = os.path.join("logs", "lifestyle_advice.txt")

    try:
        if category == "medical":
            result = Crew(
                agents=[medical_researcher],
                tasks=[research_task],
                verbose=False
            ).kickoff(inputs={'query': query})

            log_med_results(med_path)
            return result
        
        elif category == "lifestyle":
            result = Crew(
                agents=[health_coach],
                tasks=[advice_task],
                verbose=False
            ).kickoff(inputs={"query": query})

            log_health_results(life_path)
            return result
        
        else:
            # Run the crew and capture full response
            result = diabetes_crew.kickoff(inputs={'query': query})
            log_med_results(med_path)
            log_health_results(life_path)
        
        logger.info("Crew execution completed")

        # Log the final combined result
        #print("\n🔚 Final Combined Output:\n", result)
        # logger.info(f"Final Combined Output:\n{result}")

        # Check and log each agent’s output file (if you're writing to txt files)

        # log_med_results(med_path)
        # log_health_results(life_path)
        # return result

    except Exception as e:
        logger.error(f"System error while processing query: {str(e)}")
        return f"System error: {str(e)}"

if __name__ == "__main__":
    test_queries = [
        "what are symptoms of type 2 diabetes?",
        "for an indian person suffering with diabetes, suggest some breakfast"
    ]

    for query in test_queries:
        response = get_diabetes_info(query)
        print("\nResponse:\n", response)