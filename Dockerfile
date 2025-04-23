# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by some Python packages (e.g., pymupdf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable (optional, can be overridden)
ENV STREAMLIT_SERVER_PORT=8501

# Run streamlit_app.py when the container launches
# Use --server.enableCORS=false if needed behind a proxy
# Use --server.enableXsrfProtection=false if needed behind a proxy
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"] 