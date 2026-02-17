FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install deps
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY api/ api/

# Copy trained artifacts
COPY artifacts/ artifacts/

RUN uv sync --frozen --no-dev

ENV MODEL_ARTIFACT_DIR=artifacts/latest
EXPOSE 8080

CMD ["uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]

