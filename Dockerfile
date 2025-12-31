FROM python:3.10-slim

WORKDIR /app

# Combine updates, install deps, pip install, and cleanup in ONE layer to save space
COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

# Run as root
# Using direct python invocation to avoid path issues with uvicorn binary
CMD ["python", "app.py"]
