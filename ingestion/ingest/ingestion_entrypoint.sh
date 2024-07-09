#!/bin/bash

# Wait for filter to complete
while [ ! -f /data/filter_complete ]; do
  echo "Waiting for filter to complete..."
  sleep 10
done

echo "Filter has completed. Starting ingestion..."

# Now run the main container command
python /src/ingest.py
