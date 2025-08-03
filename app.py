import streamlit as st
from main import get_diabetes_info
from utils.logger import init_logging
import os

# Adding below 3 imports to fix error while deploying on streamlit cloud
# import sys
# import pysqlite3
# sys.modules["sqlite3"] = pysqlite3

# Init logs directory
logger = init_logging("streamlit.log")

# App config
st.set_page_config(page_title="Diabetes FAQ Chatbot", page_icon="🧬", layout="centered")
st.title("🩺 Diabetes FAQ Chatbot")
st.markdown("""
Ask any diabetes-related question. The assistant will pull medical facts from trusted sources
and provide practical health coaching advice using real-time web search.
""")

# Input
user_query = st.text_input("Enter your question:", placeholder="e.g. What are the early signs of type 2 diabetes?")

if user_query:
    with st.spinner("Analyzing with expert agents..."):
        try:
            result = get_diabetes_info(user_query)
            st.success("✅ Here's what we found:")
            st.markdown(result)

            with st.expander("📄 View Raw Logs"):
                if os.path.exists("logs/medical_research.txt"):
                    with open("logs/medical_research.txt") as f:
                        st.markdown("**🧠 Medical Research Agent Output:**")
                        st.code(f.read(), language='markdown')

                if os.path.exists("logs/lifestyle_advice.txt"):
                    with open("logs/lifestyle_advice.txt") as f:
                        st.markdown("**🏋️ Health Coach Agent Output:**")
                        st.code(f.read(), language='markdown')

        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.info("💡 Enter a question related to diabetes to get started.")
