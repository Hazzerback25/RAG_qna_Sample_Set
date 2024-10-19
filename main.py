from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder)
from langchain.chains import ConversationChain
from qdrant_client import QdrantClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.http.models import VectorParams, PointStruct, HnswConfig
import streamlit as st
from streamlit_chat import message
from langchain.embeddings import OllamaEmbeddings
import os
import tempfile
from utils.utils import *

# Qdrant client
client = QdrantClient(url="http://localhost:6333") 

#Streamlit App
st.subheader("PDF Q&A using RAG")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# llm = Ollama(base_url='http://host.docker.internal:11434',model="llama3.2:1b")
llm = Ollama(model="llama3.2:1b")

# import time

def process_pdf(uploaded_file):
    # Use a fixed collection name
    collection_name = "rag_embeddings"
    st.session_state['collection_name'] = collection_name  # Store in session state

    # Write the uploaded PDF file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name  # Get the path to the temporary file
    # Load and process the documents
    docs = load_documents(temp_file_path)
    split_docs = split_documents(docs)
    embeddings = create_embeddings(split_docs)
    # Insert embeddings in Qdrant with the collection name
    insert_embeddings_in_qdrant(client, embeddings, split_docs, collection_name)
    os.remove(temp_file_path)

# Streamlit UI for uploading PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    with st.spinner("Processing PDF..."):
        process_pdf(uploaded_file)
    st.success("PDF processed and data stored in Qdrant!")

# conversation set up
system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# Container for chat history
response_container = st.container()
# Container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("Generating response..."):
            conversation_string = " ".join(st.session_state['responses'])  # Concatenate all responses for context
            refined_query = query_refiner(conversation_string, query)
            
            # Create query embedding
            query_embedding = OllamaEmbeddings(model="bge-large").embed_query(refined_query)
            
            # Retrieve relevant context from the most recent collection
            relevant_texts = get_relevant_context(query_embedding, client, st.session_state['collection_name'])  # Access collection_name from session state
            
            # Prepare context
            context = prepare_context(relevant_texts)
            
            # Use the conversation chain to generate a response
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            
            # Append the query and response to the session state for display
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
