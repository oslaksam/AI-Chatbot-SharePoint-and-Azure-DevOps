FROM python:3.11-slim-bullseye

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the contents of the ingest directory to /ingest in the container
COPY ingest/ /src

RUN chmod +x /src/ingestion_entrypoint.sh

ENTRYPOINT ["/src/ingestion_entrypoint.sh"]