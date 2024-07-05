# Diabetes-FAQ-Chatbot

This project implements a FAQ chatbot for answering questions related to diabetes using Streamlit and Hugging Face models.

## Overview

The Diabetes FAQ Chatbot allows users to ask questions about diabetes and receive responses based on a predefined knowledge base. It utilizes Streamlit for the web interface and Hugging Face models for natural language processing.

## Features

- **Conversation Tab**: Users can input questions, and the chatbot responds based on the context of the conversation.
- **Database Tab**: Displays information about the loaded database file used by the chatbot.
- **Chat History Tab**: Shows the history of interactions between the user and the chatbot.
- **Error Handling**: Handles exceptions such as file loading errors and unexpected exceptions gracefully.

## Installation

1. Clone the repository:

   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:

```bash
streamlit run llm_chatbot.py
```

## Project Structure

- **llm_chatbot.py**: Main application script containing the Streamlit app code.
- **chatbot.py**: Python script containing the chatbot logic.
- **requirements.txt**: List of Python dependencies for the project.

## Dependencies

- Streamlit
- Transformers (Hugging Face)
- LangChain (or any other relevant libraries used for document loading, embeddings, etc.)

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your suggested changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Feel free to customize this README file further based on additional features, instructions, or specific details about your implementation.
