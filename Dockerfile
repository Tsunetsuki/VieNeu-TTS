FROM python:3.12

WORKDIR /code

RUN apt-get update && apt-get install -y \
    espeak \
    && rm -rf /var/lib/apt/lists/*
# RUN apk add --no-cache espeak

# COPY requirements.txt requirements.txt
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# RUN pip install --no-cache-dir -r requirements.txt

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY pyproject.toml uv.lock .
ENV UV_HTTP_TIMEOUT=24000
ENV UV_CONCURRENCY=1

RUN uv sync
ENV HF_HOME="/root/.cache/huggingface"


COPY . .

EXPOSE 8000

CMD ["uv", "run", "fastapi", "dev", "endpoint.py", "--host", "0.0.0.0"]

