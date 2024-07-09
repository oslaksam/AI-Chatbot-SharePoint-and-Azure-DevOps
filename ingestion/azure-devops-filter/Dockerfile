# Use the specified base image
FROM python:3.11-slim-bullseye

# Update the package list and install curl
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /

COPY templates /templates

# Copy the rest of the code
COPY . /

# Copy the rest of the code
COPY filter_entrypoint.sh /filter_entrypoint.sh

RUN chmod +x /filter_entrypoint.sh

ENTRYPOINT ["/filter_entrypoint.sh"]