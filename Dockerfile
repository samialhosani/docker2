# Use a slim, stable version of Python
FROM python:3.14-slim

# Set environment variables to prevent Python from writing .pyc files 
# and to ensure stdout is logged immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
# build-essential is required for some C-based Python packages (like sqlite-vec)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all python files and config
COPY *.py /app/
COPY config.json* /app/

# Copy the data and materials directories
COPY data/ /app/data/
COPY materials/ /app/materials/

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Accept API key at runtime (not baked into image)
ENV GROQ_API_KEY=""
ENV OPENAI_API_KEY=""
ENV ANTHROPIC_API_KEY=""

# Command to run when the container starts:
# 1. Run data.py to convert CSVs to education_platform.db
# 2. Start the FastAPI server on 0.0.0.0 so it is accessible from outside the container
CMD sh -c "python data.py && uvicorn api:app --host 0.0.0.0 --port 8000"
