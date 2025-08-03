# app_streamlit.py
# Developed by MOAZ

import streamlit as st
from legacy_chatbot_logic import load_doc, create_db, create_chain, process_chat
from langchain.schema import AIMessage, HumanMessage

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'db' not in st.session_state:
    st.session_state['db'] = None

if 'chain' not in st.session_state:
    st.session_state['chain'] = None

def main():
    st.title("Diabetes FAQ Chatbot")
    st.write("Ask your questions about diabetes.")

    # Tabs
    tab1, tab2, tab3 = st.columns(3)

    # Load data and create chain
    DB_FILE = 'diabetes_faq.pdf'
    docs = load_doc(DB_FILE)

    if docs:
        st.session_state['db'] = create_db(docs)
        st.session_state['chain'] = create_chain(st.session_state['db'])

        with tab1:
            st.write("### Conversation")
            user_input = st.text_input("Enter your question:")
            if user_input:
                response = process_chat(st.session_state['chain'], user_input, st.session_state['chat_history'])
                st.session_state['chat_history'].append(HumanMessage(content=user_input))
                st.session_state['chat_history'].append(AIMessage(content=response['answer']))
                st.write(f"**You:** {user_input}")
                st.write(f"**Bot:** {response['answer']}")

        with tab2:
            st.write("### Database")
            if st.session_state['db']:
                st.write(f"Database Name: {DB_FILE}")
            else:
                st.write("No database loaded.")

        with tab3:
            st.write("### Chat History")
            if st.session_state['chat_history']:
                for message in st.session_state['chat_history']:
                    if isinstance(message, HumanMessage):
                        st.write(f"**You:** {message.content}")
                    elif isinstance(message, AIMessage):
                        st.write(f"**Bot:** {message.content}")
            else:
                st.write("No chat history.")
    else:
        st.error("Failed to load documents.")

if __name__ == "__main__":
    main()
