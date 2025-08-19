import hashlib
import logging
import time
from pathlib import Path
from flask import Flask, request, jsonify
import whisper
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from pydub import AudioSegment
from pypdf import PdfReader
import json
import threading
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)
CORS(app)
# Initialize vectorstore (loaded once to avoid reading PDF every time)
vectorstore = None
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
# Track start time
start_time = time.time()

def elapsed_time():
    """Calculate elapsed time in minutes."""
    elapsed = time.time() - start_time
    return f"{elapsed / 60:.2f} minutes"

def initialize_vectorstore():
    global vectorstore
    vectorstore_folder = Path("faiss_doc")
    vectorstore_folder.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

    try:
        # Attempt to load the vector store from disk
        vectorstore = FAISS.load_local(vectorstore_folder, embeddings=embeddings, allow_dangerous_deserialization=True)
        print(f"[{elapsed_time()}] Loaded vectorstore from disk....Go on")
    except Exception as e:
        # If loading fails, log the error and create a new vectorstore
        print(f"[{elapsed_time()}] Error loading vectorstore: {e}. Creating a new one...")
        threading.Thread(target=create_vectorstore).start()  # Handle creation in a separate thread

@app.route('/chat', methods=['POST'])
def chat():
    """Handle POST requests to the Flask API for chatbot."""
    user_question = request.json.get('question')

    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    
    if vectorstore is None:
        initialize_vectorstore()

    response = handle_userinput(user_question)
    return jsonify(response)

def get_text_from_pdf_files(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None return values
    return text

def get_text_chunks(text):
    """Split the text into smaller chunks for embedding."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=200,
        length_function=lambda x: min(len(x), 1000)
    )
    chunks = text_splitter.split_text(text)
    return chunks
def get_conversation(vectorstore):
    """Create a conversational retrieval chain."""
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = OllamaLLM(model="llama3.2:1b", temperature=0.1, max_tokens=100, callbacks=callback_manager)

    prompt = PromptTemplate.from_template(
        """
        You are a helpful AI assistant.
        You are trained on user guides and FAQs for products of Airbus.
        Answer the question based on the context below. If you 
        don't know the answer, just say that you don't know, don't try to make up an 
        answer.

        {context}

        Question: {question}
        Answer
        """
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return conversation_chain


def create_vectorstore():
    """Create a vector store using Ollama embeddings."""
    print(f"[{elapsed_time()}] Reading Pdf...")
    raw_text = read_pdfs()  # Read all PDFs
    print(f"[{elapsed_time()}] Reading Chunks...")
    text_chunks = get_text_chunks(raw_text)  # Get text chunks
    print(f"[{elapsed_time()}] Creating vector it will take some time...")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore_folder = Path("faiss_doc")
    vectorstore.save_local(vectorstore_folder)
    print(f"[{elapsed_time()}] Vector created and saved successfully!")

def handle_userinput(user_question):
    """Handle user input and display chat history."""
    conversation_chain = get_conversation(vectorstore)
    response = conversation_chain.invoke({"question": user_question})
    chat_history = response['chat_history']
    
    output = [{"role": "AI" if isinstance(message, AIMessage) else "Human", "content": message.content} for message in chat_history]
    
    return {"chat_history": output}

def read_pdfs():
    """Read PDF files from a folder."""
    pdf_search = Path("data/").glob("*.pdf")
    pdf_files = [str(file.absolute()) for file in pdf_search]
    if not pdf_files:
        pdf_search = Path("training/").glob("*.pdf")
        pdf_files = [str(file.absolute()) for file in pdf_search]
    return get_text_from_pdf_files(pdf_files)

if __name__ == '__main__':
    # Initialize vectorstore on app startup
    initialize_vectorstore()
    app.run(debug=True)