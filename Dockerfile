FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Generate real UCI-derived artifacts during image build.
# If a platform blocks outbound network during build, run python src/train.py locally,
# include model artifacts in the deployment build context, or use the local Kubernetes path.
RUN python -m src.data_acquisition && python -m src.train

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
