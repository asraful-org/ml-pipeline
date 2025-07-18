# inference_api/Dockerfile

FROM python:3.9-slim-buster

WORKDIR /app

COPY inference_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference_api/app.py .

EXPOSE 5001

# These are placeholders. Values will be passed via GitHub Actions 'env' or ACI environment-variables
ENV AZURE_SUBSCRIPTION_ID=""
ENV AZURE_RESOURCE_GROUP_NAME=""
ENV AZURE_ML_WORKSPACE_NAME=""
ENV REGISTERED_MODEL_NAME="IrisLogisticRegressionModel"
ENV MODEL_VERSION="latest"

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]