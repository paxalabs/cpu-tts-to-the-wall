FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng procps && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 && \
    pip install --no-cache-dir onnxruntime==1.20.1 openvino==2026.3.0 numpy soundfile \
        "kokoro>=0.9" "misaki[en]>=0.9" psutil
WORKDIR /work
