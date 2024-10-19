###Project Architecture:
https://excalidraw.com/#json=zH4U_Avw9GlLr4HCJ2Bla,4BKgttGcGNdvUJyCg--TBw
## Setup

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

### Running with Docker:
- comment out 'llm = Ollama(model="llama3.2:1b")' and uncomment 'llm = Ollama(base_url='http://host.docker.internal:11434',model="llama3.2:1b")' in main.py
- Build the docker image:
    ```bash
    docker build -t rag_app .
    ```
- Run the docker container:
  ```bash
    docker run -p 8501:8501 rag_app 
    ```
#### Running without Docker:

- In the terminal while staying on 'RAG_qna_Sample_Set' run 
    ```bash
    streamlit run main.py
    ```

  Expected Output:
<img width="1427" alt="Screenshot 2024-10-19 at 12 17 05 PM" src="https://github.com/user-attachments/assets/57d825b9-c804-4df6-aca5-b38bff30104b">


#### Testing instructions:
There are 3 tests included in test_app.py, run them using:
    ```bash
    pytest test_app.py 
    ```


### Tech Used:
- Python, Langchain, Ollama, Docker, Qdrant(Vector Database), llama3.2 (LLM)

### Project Details:
- main.py: Contains the streamlit frontend code.
- utils.py: Contains the user-defined functions to be used by streamlit components.
- main.ipynb: Notebook demonstrating the RAG model creation and implementation.
    
