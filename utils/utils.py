from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from qdrant_client.http.models import VectorParams, PointStruct, HnswConfig
from langchain_community.llms import Ollama

#Contains user defined functions to be used for streamlit UI in main.py
def load_documents(file):
    loader = PyPDFLoader(file)
    docs = loader.load()
    return docs

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', ' ', '', "(?<=\.)"]
    )
    return text_splitter.split_documents(documents)

def create_embeddings(documents):
    embedding_model = OllamaEmbeddings(model="bge-large")
    return embedding_model.embed_documents([doc.page_content for doc in documents])

# def create_or_update_collection(client, collection_name, embedding_size):
#     if client.collection_exists(collection_name=collection_name):
#         print(f"Collection '{collection_name}' already exists. Clearing existing data.")
#         client.delete_collection(collection_name)  # Clear existing data if the collection exists

#     print(f"Creating collection: {collection_name}")
#     client.create_collection(
#         collection_name=collection_name,
#         vectors_config=VectorParams(size=embedding_size, distance="Cosine"),
#         hnsw_config=HnswConfig(m=16, ef_construct=200, full_scan_threshold=1000).dict()
#     )
        
def create_or_update_collection(client, collection_name, embedding_size):
    # Check if the collection exists by listing all collections
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if collection_name in collection_names:
        print(f"Collection '{collection_name}' already exists. Clearing existing data.")
        client.delete_collection(collection_name)  # Clear existing data if the collection exists

    print(f"Creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_size, distance="Cosine"),
        hnsw_config=HnswConfig(m=16, ef_construct=200, full_scan_threshold=1000).dict()
    )



def insert_embeddings_in_qdrant(client, embeddings, documents, collection_name="rag_embeddings", batch_size=1000):
    # Ensure the collection is created or cleared
    create_or_update_collection(client, collection_name, len(embeddings[0]))
    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={"text": documents[i].page_content, "metadata": documents[i].metadata}
        ) for i in range(len(embeddings))
    ]

    # Upsert points into the collection
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[i:i + batch_size])

def get_relevant_context(query_embedding, client, collection_name="rag_embeddings", top_k=5):
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )
    return [result.payload['text'] for result in search_results]


def query_refiner(conversation_string, query):
    llm = Ollama(model="llama3.2:1b")  # Initialize the model

    # Create a prompt for the model
    prompt = f"""
    Given the following conversation log and user query, refine the query to make it more relevant to provide an answer from a knowledge base.

    CONVERSATION LOG:
    {conversation_string}

    User Query: {query}

    Refined Query:
    """

    # Generate a refined query using the LLM
    refined_query = llm.predict(prompt)  # Use the model to generate the response
    return refined_query

def prepare_context(relevant_texts):
    return "\n\n".join(relevant_texts)