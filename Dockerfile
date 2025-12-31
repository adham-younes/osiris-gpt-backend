FROM python:3.10-slim

WORKDIR /app

# Combine updates, install deps, pip install, and cleanup in ONE layer to save space
# We install build-essential for compilation, but we don't remove it to be safe, 
# just clean the apt cache.
COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

# Run as root (simplest for Spaces) to avoid permission issues and extra layers
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
