# Dynamoscope · production-ready CPU image
# Build : docker build -t dynamoscope .
# Run   : docker run -p 8770:8770 -v $(pwd)/videos:/app/videos dynamoscope

FROM python:3.13-slim AS base

# System deps : ffmpeg for audio extract + opencv runtime + git for torch.hub
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast deps install
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy + install deps separately for layer caching
COPY pyproject.toml /app/
COPY backend/__init__.py /app/backend/
RUN uv venv --python 3.13 && uv pip install -e .

# Copy code
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Create writable dirs
RUN mkdir -p /app/cache /app/videos

ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV HF_HOME=/app/cache/hf
ENV TORCH_HOME=/app/cache/torch

EXPOSE 8770

CMD ["/app/.venv/bin/python", "-m", "uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", "--port", "8770", "--log-level", "info"]
