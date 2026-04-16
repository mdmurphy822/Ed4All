FROM python:3.11-slim

# System dependencies for DART PDF processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[full]" 2>/dev/null || pip install --no-cache-dir -e .

# Copy project
COPY . .
RUN pip install --no-cache-dir -e ".[full]"

# Default: start MCP server
CMD ["python", "-m", "MCP.server"]
