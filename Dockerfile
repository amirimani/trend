# Live BTC 4h signal monitor -> Telegram alerts
FROM python:3.11-slim

# Faster, quieter Python in containers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

WORKDIR /app

# Install runtime deps first for better layer caching
COPY requirements-live.txt .
RUN pip install --no-cache-dir -r requirements-live.txt

# App code (only what the service needs)
COPY src ./src

# Persisted dedup state lives here (mount a volume)
RUN mkdir -p /data
ENV STATE_FILE=/data/state.json

# Drop privileges
RUN useradd -m -u 10001 appuser && chown -R appuser /app /data
USER appuser

# Lightweight liveness check: state file is refreshed each closed 4h bar.
HEALTHCHECK --interval=10m --timeout=10s --start-period=2m --retries=3 \
    CMD python -c "import os,sys; sys.exit(0 if os.path.exists(os.getenv('STATE_FILE','/data/state.json')) else 1)"

CMD ["python", "-m", "src.live.monitor"]
