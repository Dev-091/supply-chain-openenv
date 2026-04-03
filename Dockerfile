FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy local files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=7860

# Expose the API port
EXPOSE 7860

# Start the environment server
# Hugging Face Spaces will use this to serve the OpenEnv API
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
