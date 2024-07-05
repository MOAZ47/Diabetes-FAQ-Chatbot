# Diabetes-FAQ-Chatbot

This project implements a chatbot using a Large Language Model (LLM) to answer frequently asked questions (FAQs) about diabetes. It leverages Streamlit for the user interface and integrates with Hugging Face transformers for natural language processing.

## Overview

The Diabetes FAQ Chatbot allows users to ask questions about diabetes and receive responses based on a predefined knowledge base. It utilizes Streamlit for the web interface and Hugging Face models for natural language processing.

## Purpose
The purpose of this project is to provide a user-friendly interface for retrieving information related to diabetes through conversational interactions. By leveraging advanced AI models, the chatbot aims to:

- **Educate**: Provide accurate and accessible information about diabetes types, symptoms, treatments, and related topics.

- **Support**: Assist users in finding answers to their questions promptly and efficiently.

- **Engage**: Foster engagement by offering a conversational interface that mimics human-like responses, enhancing user experience.

## Features

- **Conversation Tab**: Users can input questions, and the chatbot responds based on the context of the conversation.
- **Database Tab**: Displays information about the loaded database file used by the chatbot.
- **Chat History Tab**: Shows the history of interactions between the user and the chatbot.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MOAZ47/diabetes-faq-chatbot.git
   cd diabetes-faq-chatbot
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the Streamlit app:

```bash
streamlit run app_streamlit.py
```

To run solely the chatbot app:

```bash
python chatbot_logic.py
```

## Project Structure

- **app_streamlit.py**: Main application script containing the Streamlit app code.
- **chatbot_logic.py**: Python script containing the chatbot logic.
- **requirements.txt**: List of Python dependencies for the project.

## Dependencies

- Streamlit
- Transformers (Hugging Face)
- LangChain

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your suggested changes.

