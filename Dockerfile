# ==============================================================================
# GuardianVision AI — Production Dockerfile
# Multi-stage build: slim runtime image, no build tools left behind.
# ==============================================================================

FROM python:3.11-slim AS base

# System dependencies required by OpenCV and Ultralytics at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Dependency layer (cached separately from source for faster rebuilds) ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Application source ----
COPY app/ ./app/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY configs/ ./configs/
COPY main.py .

# Runtime directories created at build time so volume mounts have a target
RUN mkdir -p models reports logs assets/screenshots assets/videos

# Non-root user for security
RUN useradd --create-home --shell /bin/bash guardian && \
    chown -R guardian:guardian /app
USER guardian

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/status')" || exit 1

# Default: run the API. Override CMD to run dashboard or pipeline instead.
CMD ["python", "main.py", "--mode", "api"]
