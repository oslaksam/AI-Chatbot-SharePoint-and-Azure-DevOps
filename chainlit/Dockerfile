# Use an appropriate base image, e.g., python:3.10-slim
FROM --platform=linux/amd64 python:3.11-slim

# Set environment variables (e.g., set Python to run in unbuffered mode)
ENV PYTHONUNBUFFERED 1

# Set the working directory
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

# Copy your application code into the container
# copy all from src folder to /app folder except the files in .dockerignore
COPY src/ /app/

RUN pip install -r /app/requirements.txt

EXPOSE 8080
EXPOSE 9000

CMD ["python", "-m", "chainlit", "run", "deployment_dev.py", "-h", "--port", "8080"]