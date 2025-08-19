import time
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
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
from pypdf import PdfReader
import json
import hashlib

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)
CORS(app)

# Initialize vectorstore (loaded once to avoid reading PDF every time)
vectorstore = None
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# Control flag to enable/disable adding interactions to vectors
ADD_INTERACTIONS_TO_VECTORS = True  # Set to False to disable adding interactions

# Track start time
start_time = time.time()

def hash_text(text):
    """Generate a unique hash for the given text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

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
        # Load user interactions and update the vector store
        load_user_interactions()
    except Exception as e:
        # If loading fails, log the error and create a new vectorstore
        print(f"[{elapsed_time()}] Creating a new one....")
        create_vectorstore()  # Handle creation in the main thread for simplicity

def load_user_interactions():
    """Load user interactions from file and add to vector store if new."""
    folder = Path("chat_history")
    interaction_file = folder / "user_interactions.txt"
    if interaction_file.exists():
        with open(interaction_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 4):  # Each interaction takes approximately 4 lines
                if i < len(lines):
                    user_question_line = lines[i].strip()
                    if user_question_line.startswith("User Question: "):
                        user_question = user_question_line.split(": ", 1)[1]
                    else:
                        continue

                if i + 1 < len(lines):
                    ai_response_line = lines[i + 1].strip()
                    if ai_response_line.startswith("AI Response: "):
                        ai_response = ai_response_line.split(": ", 1)[1]
                    else:
                        continue

                user_answer = None
                if i + 2 < len(lines):
                    user_answer_line = lines[i + 2].strip()
                    if user_answer_line.startswith("User Answer: "):
                        user_answer = user_answer_line.split(": ", 1)[1]

                # Add the user question, AI response, and user answer to the vector store if they are new
                if ADD_INTERACTIONS_TO_VECTORS:
                    add_if_new_to_vector_store(user_question)
                    add_if_new_to_vector_store(ai_response)
                    if user_answer:
                        add_if_new_to_vector_store(user_answer)


def add_if_new_to_vector_store(text):
    """Add text to the vector store if it's not already there (using a hash to check for duplicates)."""
    text_hash = hash_text(text)  # Get the hash of the text
    if text_hash in added_text_hashes:
        print(f"[{elapsed_time()}] Duplicate detected. Text not added: {text}")
        return  # Skip adding if the hash is already in the set

    try:
        # Check if the text is already in the vector store using similarity search
        if not vectorstore.similarity_search(text):  # If no similar text exists, add it
            print(f"[{elapsed_time()}] Adding to vector store: {text}")
            vectorstore.add_texts(texts=[text], embedding=embeddings)
            added_text_hashes.add(text_hash)  # Track this text as added
        else:
            print(f"[{elapsed_time()}] Already added to vector store based on similarity search.")
    except Exception as e:
        print(f"[{elapsed_time()}] Error during vector store operation: {e}")
        print(f"[{elapsed_time()}], Adding text anyway as it is not found in the vector store: {text}")
        vectorstore.add_texts(texts=[text], embedding=embeddings)
        added_text_hashes.add(text_hash)

# Global set to track the hashes of texts that have already been added to the vector store
added_text_hashes = set()

@app.route('/chat', methods=['POST'])
def chat():
    """Handle POST requests to the Flask API for chatbot."""
    user_question = request.json.get('question')
    user_answer = request.json.get('answer')  # New input for user-provided answer

    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    
    if vectorstore is None:
        initialize_vectorstore()

    response = handle_userinput(user_question, user_answer)
    return jsonify(response)

def get_text_from_pdf_files(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None return values
    return text
def get_text_from_txt_files(txt_files):
    text = ""
    for txt in txt_files:
        text += open(txt).read()
    return text

def read_txt_files():
    # Read all txt files in data folder
    text_search = Path("chat_history/").glob("*.txt")
    txt_files = [str(file.absolute()) for file in text_search]
    # get text from txt files
    transcript_text = get_text_from_txt_files(txt_files) + " "
    return transcript_text

def get_text_chunks(text):
    """Split the text into smaller chunks for embedding."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=lambda x: min(len(x), 1000)
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_conversation(vectorstore):
    """Create a conversational retrieval chain."""
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = OllamaLLM(model="llama3.2:1b", temperature=0.4, max_tokens=50, callbacks=callback_manager)

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
    raw_text = read_pdfs()  # Read all PDFs
    text_chunks = get_text_chunks(raw_text)  # Get text chunks
    print(f"[{elapsed_time()}] Creating vector...")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore_folder = Path("faiss_doc")
    vectorstore.save_local(vectorstore_folder)
    print(f"[{elapsed_time()}] Vector created and saved successfully!")


def handle_userinput(user_input, user_answer=None):
    """Handle user input and display chat history."""
    
    # Check if the user input contains context
    if user_input.startswith("Context:"):
        context, question = user_input.split("Question:", 1)
        context = context.replace("Context:", "").strip()
        question = question.strip()

        # Optionally save the context to the vector store
        if ADD_INTERACTIONS_TO_VECTORS:
            add_if_new_to_vector_store(context)
    else:
        question = user_input

    conversation_chain = get_conversation(vectorstore)
    response = conversation_chain.invoke({"question": question})
    chat_history = response['chat_history']
    
    output = [{"role": "AI" if isinstance(message, AIMessage) else "Human", "content": message.content} for message in chat_history]
    
    # Save the user question and AI response for learning
    save_interaction(question, output[-1]['content'], user_answer)  # Save interaction without user answer

    return {"chat_history": output}

def save_interaction(user_question, ai_response, user_answer=None):
    """Save user interaction for future learning."""
    Path("chat_history").mkdir(parents=True, exist_ok=True)  # Ensure the folder exists
    with open('chat_history/user_interactions.txt', 'a', encoding='utf-8') as f:
        f.write(f"User Question: {user_question}\nAI Response: {ai_response}\n")
        if user_answer:
            f.write(f"User Answer: {user_answer}\n")
        f.write("\n")

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