#from langchain_community.document_loaders.csv_loader import CSVLoader              # CSV LOADER
from langchain.document_loaders import PyPDFLoader                                  # PDF Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter                  # chunking
from langchain.vectorstores import FAISS                                            # vector DB
from langchain.embeddings import HuggingFaceEmbeddings                              # embedding
from langchain_huggingface import HuggingFacePipeline                               # pipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder          # template, placeholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain # retriever
from langchain.chains.combine_documents import create_stuff_documents_chain         # retriever
from langchain.schema import AIMessage, HumanMessage

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline              # model, tokenizer, pipeline


def load_doc(file):
    try:
        # Load the PDF file
        loader = PyPDFLoader(file)
        documents = loader.load()

        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunked_docs = text_splitter.split_documents(documents)

        return chunked_docs
    except FileNotFoundError as e:
        print(f"File not found: {str(e)}")
        return None
    except OSError as e:
        print(f"OS error: {str(e)}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None

def create_db(docs):
    # Using HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

def create_chain(vectorstore):
    model_name = "EleutherAI/gpt-neo-125M"
    _model = AutoModelForCausalLM.from_pretrained (model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("text-generation", model = _model, tokenizer = tokenizer, max_new_tokens = 512)
    model = HuggingFacePipeline(pipeline = pipe)

    # Connect query to FAISS index using a retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 4})
    
    qa_system_prompt = """You are an assistant for medical question-answering tasks. \
    You are designed specifically to answer questions related to Diabetes.\
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\
    {context}"""
    prompt = ChatPromptTemplate.from_messages([
        ('system', qa_system_prompt),
        MessagesPlaceholder(variable_name = "chat_history"),
        ('human', "{input}")
    ])

    chain = create_stuff_documents_chain(
        llm = model,
        prompt = prompt
    )
    
    retriever_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name = "chat_history"),
        ('human', "{input}"),
        ('system', retriever_system_prompt)
    ])

    history_aware_retriever = create_history_aware_retriever(model, retriever, retriever_prompt)

    retrival_chain = create_retrieval_chain(history_aware_retriever, chain)
    return retrival_chain

def process_chat(chain, question, chat_history):
    response = chain.invoke({
        'input': question,
        'chat_history': chat_history
        })
    return response

if __name__ == '__main__':
    docs = load_doc('diabetes_faq.pdf')
    db = create_db(docs)
    chain = create_chain(db)

    chat_history = []

    while True:
        user_input = input("Enter your question: ")

        if user_input.lower() == 'exit':
            break

        response = process_chat(chain, user_input, chat_history)

        chat_history.append(HumanMessage(content = user_input))
        chat_history.append(AIMessage(content = response['answer']))

        #print(f"Response part is: {response}")
        print(f"**Bot:** {response['answer']}")
