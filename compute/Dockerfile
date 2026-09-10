# Fashion Police — compute tier.
# Runs on the CUDA host (e.g. an NVIDIA DGX Spark: ARM64 + Grace-Blackwell GB10).
# Use the datacenter NGC PyTorch image (the "-igpu" tags are Jetson/Orin builds,
# compiled only for sm_87 and will not run on GB10 / sm_121). Bump BASE_IMAGE as
# NGC releases newer builds.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    FP_COMPUTE_STYLES_PATH=/config/styles.yaml \
    FP_COMPUTE_DEVICE=auto

# torch / torchvision are already in the base image — install only the rest.
RUN pip install --no-cache-dir \
      "fastapi>=0.115" "uvicorn[standard]>=0.32" "python-multipart>=0.0.12" \
      "pydantic>=2.9" "pydantic-settings>=2.6" "pyyaml>=6.0" "pillow>=10.4" \
      "transformers>=4.44" "huggingface-hub>=0.24" "safetensors>=0.4.4"

WORKDIR /srv
COPY app ./app

EXPOSE 8801

# start-period is generous: first boot downloads ~700 MB of weights into /models.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=10 \
  CMD python -c "import sys,urllib.request,json; sys.exit(0 if json.load(urllib.request.urlopen('http://localhost:8801/healthz'))['ready'] else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8801"]
