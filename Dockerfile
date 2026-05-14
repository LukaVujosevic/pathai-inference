ARG BASE_IMAGE=pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel
FROM ${BASE_IMAGE}

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_BACKEND=mock \
    HOST=0.0.0.0 \
    PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libglib2.0-0 \
        libgl1 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY requirements-gpu.txt .
ARG INSTALL_GPU_DEPS=false
RUN if [ "$INSTALL_GPU_DEPS" = "true" ]; then \
        python -m pip install -r requirements-gpu.txt; \
    fi

COPY app ./app
COPY configs ./configs
COPY scripts ./scripts
COPY README.md .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
