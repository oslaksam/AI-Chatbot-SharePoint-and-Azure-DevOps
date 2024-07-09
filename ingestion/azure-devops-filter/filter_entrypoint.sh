#!/bin/bash

# Wait for scrape to complete
while [ ! -f /data/scrape_complete ]; do
  echo "Waiting for scrape to complete..."
  sleep 10
done

echo "Scrape has completed. Starting filter..."

# Now run the main container command
python cleanup.py
