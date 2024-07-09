# Use the specified base image
FROM python:3.11-slim-bullseye

# Update the package list and install curl
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*


# Copy the requirements file into the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . /

# Set the default command to run your script
CMD ["python", "ado_scrape.py"]
