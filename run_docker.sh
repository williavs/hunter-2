#!/bin/bash

# Stop any existing container with the same name
docker stop gtm-scrape-container 2>/dev/null || true
docker rm gtm-scrape-container 2>/dev/null || true

# Build the Docker image
echo "Building Docker image..."
docker build -t gtm-scrape-image .

# Run the Docker container
echo "Running Docker container..."
docker run -p 8501:8501 --name gtm-scrape-container gtm-scrape-image

# Note: Press Ctrl+C to stop the container 