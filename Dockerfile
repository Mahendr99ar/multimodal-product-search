# CPU inference image for the search API.
# Build:  docker build -t product-search .
# Run:    docker run -p 8000:8000 -v $PWD/artifacts:/app/artifacts product-search
#   or pull artifacts from S3 at startup:
#         docker run -p 8000:8000 -e ARTIFACTS_S3_URI=s3://<bucket>/product-search/ \
#                -e AWS_DEFAULT_REGION=ap-south-1 product-search
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.cache

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the base OpenCLIP weights into the image so containers start without internet.
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')"

COPY src/ src/
COPY serve/ serve/

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
