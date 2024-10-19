
FROM python:3.10-slim


WORKDIR /app


RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /app/requirements.txt


RUN pip install --no-cache-dir -r requirements.txt


COPY . /app


CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
