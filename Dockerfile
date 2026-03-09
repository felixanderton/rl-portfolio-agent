FROM --platform=linux/amd64 pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps required by stable-baselines3[extra] (opencv headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]
