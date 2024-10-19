## Setup

### Running locally

#### Running without Docker:

- Clone the repository and navigate to the 'RAG_qna_Sample_Set'

- Load the environment
    ```bash
  pipenv shell
    ```

    ```bash
  pipenv install -r requirements.txt
    ```

- Setup Qdrant locally using Docker and run it on localhost:6333
    https://qdrant.tech/documentation/quickstart/

- In the terminal while staying on 'RAG_qna_Sample_Set' run 
    ```bash
    streamlit run main.py
    ```