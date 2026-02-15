# Backend: FastAPI application server
# Build context is the project root (requirements.txt lives there).
# Run: docker build -f backend/Dockerfile .

FROM python:3.12-slim-bookworm

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

# Install system deps required by opencv-python-headless and health check
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Install Python deps (copy requirements first for layer caching).
# Filter out test-only packages (pytest, httpx) to keep the image lean.
COPY requirements.txt .
RUN grep -vE '^\s*(pytest|httpx)' requirements.txt > requirements-prod.txt && \
    pip install --no-cache-dir -r requirements-prod.txt && \
    rm requirements-prod.txt && \
    pip cache purge

# Copy backend package
COPY --chown=appuser:appuser backend/ backend/

# Create data directories with correct ownership
RUN mkdir -p /app/datasets /app/models /app/runs && \
    chown -R appuser:appuser /app/datasets /app/models /app/runs

# Switch to non-root user
USER appuser

# Volumes for persistent data
VOLUME ["/app/datasets", "/app/models", "/app/runs"]

EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Health check against the FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
