FROM python:3.11-slim

WORKDIR /app

# Upgrade system and install build deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY ./requirements.txt /app/requirements.txt

# Install python deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Fix permissions
RUN chmod -R 777 /app

# Explicitly expose port 7860
EXPOSE 7860

# Use python module execution for uvicorn to ensure path correctness
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
