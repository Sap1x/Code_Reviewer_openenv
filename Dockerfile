# ─────────────────────────────────────────────────────
# CodeRL — Agentic Code Review RL Environment
# Production Docker image for Hugging Face Spaces
# ─────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Install dependencies ──────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────
COPY . .

# ── Create non-root user (HF Spaces requirement) ─────
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# ── Expose port (HF Spaces default: 7860) ────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# ── Run server ────────────────────────────────────────
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
