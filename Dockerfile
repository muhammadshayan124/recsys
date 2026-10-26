FROM python:3.12-slim AS build
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -e .
# The dataset cannot be redistributed, so the model is built from the official download at image build time.
COPY results/tuning.json ./results/tuning.json
RUN recsys fetch && recsys export && rm -rf data

FROM build AS serve
EXPOSE 8000
CMD ["uvicorn", "recsys.serve:app", "--host", "0.0.0.0", "--port", "8000"]
