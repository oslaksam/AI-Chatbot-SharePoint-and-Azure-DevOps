# Use an updated Python runtime as a parent image, based on Debian Bullseye
FROM python:3.11-slim-bookworm

# Set the working directory in the container to /app
WORKDIR /app

# Install system dependencies for OpenCV and potentially newer sqlite3
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    netcat-openbsd \
    libglib2.0-0 \
    curl \
    libreoffice \
    mesa-utils && \
    rm -rf /var/lib/apt/lists/*
# Copy only requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Add the current directory contents into the container at /app
COPY ./ingest.py /app/app.py
COPY ./sharepoint_filter.py /app/sharepoint_filter.py
COPY ./wait_for_chromadb.sh /app/wait_for_chromadb.sh

RUN chmod +x /app/wait_for_chromadb.sh

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
# Run wait_for_chromadb.sh to wait for ChromaDB to be ready before starting the app
CMD ["python", "app.py"]
