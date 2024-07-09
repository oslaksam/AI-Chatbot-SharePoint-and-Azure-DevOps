# Use an updated Python runtime as a parent image, based on Debian Bullseye
FROM python:3.11-slim-bookworm

# Set the working directory in the container to /app
WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Add the current directory contents into the container at /app
COPY ./scrape_sharepoint.py /app/app.py

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
# Run wait-for.sh to wait for ChromaDB to be ready before starting the app
CMD ["python", "app.py"]
