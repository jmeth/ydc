# YOLO Dataset Creator
# For local/personal use only - not vetted for production deployments

FROM python:3.12-slim-bookworm AS base

# Security: Don't run as root
# Create non-root user early
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

# Security: Install security updates and remove unnecessary packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy application code
COPY --chown=appuser:appuser server.py .
COPY --chown=appuser:appuser static/ static/

# Create datasets directory with correct permissions
RUN mkdir -p /app/datasets && chown appuser:appuser /app/datasets

# Security: Switch to non-root user
USER appuser

# Volume for persistent dataset storage
VOLUME ["/app/datasets"]

# Expose the application port
EXPOSE 5001

# Environment variables
ENV FLASK_ENV=production \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Healthcheck - verify the server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:5001/ || exit 1

# Run the application
# Using flask directly for production (gunicorn would be better for real production)
CMD ["python", "server.py"]
