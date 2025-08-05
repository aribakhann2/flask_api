# Use Python slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for ML libraries)
RUN apt-get update && apt-get install -y \
    gcc libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app
COPY . .

# Set environment variable for Flask
ENV PORT=8080

# Run Flask with Gunicorn for production
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers=2", "--threads=4", "--timeout=1200"]
