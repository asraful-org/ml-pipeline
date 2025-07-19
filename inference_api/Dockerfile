# inference_api/Dockerfile

FROM python:3.9-slim-buster

WORKDIR /app

COPY inference_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference_api/app.py .
COPY inference_api/model.pkl .

EXPOSE 5001

# These are placeholders. Values will be passed via GitHub Actions 'env' or ACI environment-variables
ENV AZURE_SUBSCRIPTION_ID=""
ENV AZURE_RESOURCE_GROUP_NAME=""
ENV AZURE_ML_WORKSPACE_NAME=""
ENV REGISTERED_MODEL_NAME="IrisLogisticRegressionModel"
ENV MODEL_VERSION="latest"


# Health check (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5001/ || exit 1

#CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]

# Use exec form for better signal handling
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--timeout", "60", "app:app"]