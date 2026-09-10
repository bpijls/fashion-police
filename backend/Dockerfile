# Fashion Police — backend gateway. Light: no GPU, no ML framework.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    FP_BACKEND_DATA_DIR=/srv/data \
    FP_BACKEND_YUNET_PATH=/models/face_detection_yunet_2023mar.onnx

# opencv-python-headless still needs glib + libgomp at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      "fastapi>=0.115" "uvicorn[standard]>=0.32" "python-multipart>=0.0.12" \
      "httpx>=0.27" "pydantic>=2.9" "pydantic-settings>=2.6" \
      "pillow>=10.4" "numpy>=1.26" "opencv-python-headless>=4.10"

WORKDIR /srv
COPY app ./app

EXPOSE 8800

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import sys,urllib.request,json; d=json.load(urllib.request.urlopen('http://localhost:8800/api/v1/health')); sys.exit(0 if d['backend']=='ok' else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8800"]
