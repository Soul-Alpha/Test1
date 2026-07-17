# Prometheus - Institutional AI Market Analysis
FROM python:3.11-slim

# Build args
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create runtime directories
RUN mkdir -p models outputs charts storage data

# Expose ports
EXPOSE 8000 8501

# Default: start API server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
