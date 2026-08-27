# ---- Stage 1: Builder ----
FROM python:3.12-slim AS builder

WORKDIR /build

# ( layer caching)
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache-dir .

# ---- Stage 2: Runtime ----
FROM python:3.12-slim AS runtime


RUN useradd --create-home --uid 1000 appuser

WORKDIR /app


COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code and model
COPY src/ ./src/
COPY --chown=appuser:appuser models/ ./models/

ENV PYTHONPATH=/app/src

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "mlops_practitioner.api.main:app", "--host", "0.0.0.0", "--port", "8000"]