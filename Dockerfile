FROM python:3.13-slim@sha256:7249d5587b78209a006dc4a3b5340678f84ea729239193aa559962cd8c1fce50

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the public, quantized BGE model into the image for offline inference.
ARG MODEL_URL=https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/ea104dacec62c0de699686887e3f920caeb4f3e3
RUN mkdir /model && python -c "import urllib.request as u; u.urlretrieve('$MODEL_URL/tokenizer.json', '/model/tokenizer.json'); u.urlretrieve('$MODEL_URL/onnx/model_quantized.onnx', '/model/model.onnx')"

COPY app app
COPY alembic.ini .
COPY migrations migrations

CMD ["sh", "-c", "alembic upgrade head && python -m app.backfill && exec uvicorn app.main:app --host 0.0.0.0 --no-access-log --port ${PORT:-8000}"]
