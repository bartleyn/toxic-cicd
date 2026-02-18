FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install deps
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY api/ api/
COPY scripts/ scripts/
COPY README.md ./

RUN uv sync --frozen --no-dev

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV MODEL_ARTIFACT_DIR=artifacts
EXPOSE 8080

ENTRYPOINT ["./entrypoint.sh"]
