FROM python:3.12-slim AS build
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# CPU-only torch from the PyTorch index (the default PyPI wheel on Linux bundles CUDA, ~2 GB)
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --default-timeout=120 --retries 5 -e .
# The dataset cannot be redistributed, so it is not in the repository: run `recsys fetch` once on the build machine,
# and the archive (checked against a pinned SHA-256 on load) is copied into the image only for the build stage.
COPY data/ml-1m.zip ./data/ml-1m.zip
COPY results/tuning.json ./results/tuning.json
RUN recsys export && rm -rf data

FROM build AS serve
EXPOSE 8000
CMD ["uvicorn", "recsys.serve:app", "--host", "0.0.0.0", "--port", "8000"]
