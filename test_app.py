# These are basic unit tests.
# Since the app interacts with external services (Qdrant, LLMs), we use 
# mocking to avoid actual network calls during testing.
# The utility functions like insert_embeddings_in_qdrant could also be
# tested, but they involve more complex interaction with the database,
#  which would require mocking the Qdrant client.


import pytest
from unittest.mock import patch, MagicMock
from utils.utils import load_documents, split_documents, create_embeddings, query_refiner, prepare_context
from qdrant_client.http.models import PointStruct


# Test 1: Testing PDF Loading and Splitting
@patch('utils.utils.PyPDFLoader')
def test_load_and_split_documents(mock_loader):
    # Mocking document loader behavior
    mock_loader.return_value.load.return_value = [
        MagicMock(page_content="Page 1 content"),
        MagicMock(page_content="Page 2 content"),
    ]

    # Load documents
    docs = load_documents("dummy.pdf")
    assert len(docs) == 2  # Ensure the documents are loaded properly

    # Test splitting documents
    split_docs = split_documents(docs)
    assert len(split_docs) > 0  # Ensure documents are split into chunks


# Test 2: Testing Embedding Creation
@patch('utils.utils.OllamaEmbeddings')
def test_create_embeddings(mock_embeddings):

    mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    
    # Create dummy documents
    docs = [MagicMock(page_content="Some content"), MagicMock(page_content="More content")]
    
    # Create embeddings
    embeddings = create_embeddings(docs)
    
    assert len(embeddings) == 2  # Ensure embeddings are created for each document

# Test 3: Test Preparing Context
def test_prepare_context():
    # Test if relevant texts are concatenated properly
    relevant_texts = ["Context 1", "Context 2", "Context 3"]
    context = prepare_context(relevant_texts)
    
    assert context == "Context 1\n\nContext 2\n\nContext 3"  # Check proper concatenation

