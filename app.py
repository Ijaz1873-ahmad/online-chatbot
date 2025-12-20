import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2
from lucknowllm import UnstructuredDataLoader, split_into_segments
# from constt import API_KEY
from lucknowllm import GeminiModel

def getAPI():
    api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")
    return api_key

# Function to read text from PDF using PdfReader
def read_pdf(file_path):
    pdf_reader = PyPDF2.PdfReader(open(file_path, "rb"))
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to initialize the model
def initialize_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, np.newaxis] * np.linalg.norm(b, axis=1))

# Function to load and split the data from PDFs
def load_and_split_data(pdf_paths):
    text_data = [read_pdf(path) for path in pdf_paths]
    chunks = []
    for text in text_data:
        chunks.extend(split_into_segments(text))
    return chunks

# Function to encode the chunks
def encode_chunks(model, chunks):
    return model.encode(chunks)

# Function to initialize the Gemini model
def initialize_gemini(API_KEY):
    return GeminiModel(api_key=API_KEY, model_name="gemini-2.5-flash")

# Function to display chat history
def display_chat():
    for query, response in st.session_state.history:
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 10px; padding: 10px; margin-bottom: 10px; word-wrap: break-word;">
                <strong>User:</strong> {query}
            </div>
            <div style="background-color: #e0ffe0; border-radius: 10px; padding: 10px; margin-bottom: 10px; word-wrap: break-word;">
                <strong>Response:</strong> {response}
            </div>
            """, unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    st.title("Domestic Violence Chatbot System")
    st.markdown("""
    ## This chatbot is built using the Retrieval-Augmented Generation (RAG) framework,
    leveraging Google's Generative AI model Gemini-1.5.

    ### How It Works

    Follow these simple steps to interact with the chatbot:

    1. **Enter Your API Key**: You'll need a Google API key for the chatbot to access Google's Generative AI models. Obtain your API key https://makersuite.google.com/app/apikey.

    2. **Ask a Question**: Ask any question related to the Domestic voilence in Pakistan, it will give you a precise answer.
    """)
    API_KEY = getAPI()
    # Initialize the model
    model = initialize_model()

    # Load and split the data from PDFs
    pdf_paths = ["p1.pdf", "p2.pdf"]
    chunks = load_and_split_data(pdf_paths)

    # Encode the chunks
    embedded_data = encode_chunks(model, chunks)

    # Initialize the Gemini model
    Gemini = initialize_gemini(API_KEY)

    # Define a relevance threshold
    threshold = 0.3  # Adjust this value based on your requirements

    # Initialize session state for conversation history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display chat history
    display_chat()

    # Input form at the bottom
    with st.form(key='query_form', clear_on_submit=True):
        query = st.text_input("Enter your query:")
        submit_button = st.form_submit_button(label='Query')

        if submit_button and query:
            # Encode the query
            embedded_query = model.encode([query])

            # Perform the similarity search
            similarities = cosine_similarity(embedded_query, embedded_data)
            top_indices = np.argsort(similarities[0])[::-1][:3]
            top_docs = [chunks[index] for index in top_indices]

            argumented_prompt =  f"You are an expert question answering system. I'll give you a question and context, and you'll return the answer. Query: {query} Contexts: {top_docs[0]} and your answers should be concise and if irrelevent query is ask print message that you do not know."
            model_output = Gemini.generate_content(argumented_prompt)
            response = model_output

            # Append the query and response to the conversation history
            st.session_state.history.append((query, response))

            # Refresh the page to display the updated chat history
            st.experimental_rerun()
            # st.query_params()
            # st.experimental_set_query_params(rerun=1)

            # query_params = st.experimental_get_query_params()
            # query_params["rerun"] = [str(int(query_params.get("rerun", [0])[0]) + 1)]
            # st.experimental_set_query_params(**query_params)

if __name__ == "__main__":
    main()
