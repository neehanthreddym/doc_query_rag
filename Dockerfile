FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy deployment requirements only
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/pdf_files data/vector_store

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000"]