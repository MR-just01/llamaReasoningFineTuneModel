# =========================================================
# CUDA RUNTIME
# =========================================================

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04


# =========================================================
# SYSTEM DEPENDENCIES
# =========================================================

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*


# =========================================================
# PYTHON CONFIGURATION
# =========================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app


# =========================================================
# PYTORCH CUDA 12.8
# =========================================================

RUN python3 -m pip install \
    --break-system-packages \
    --no-cache-dir \
    torch==2.10.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128


# =========================================================
# APPLICATION DEPENDENCIES
# =========================================================

COPY requirements.txt .

RUN python3 -m pip install \
    --break-system-packages \
    --no-cache-dir \
    -r requirements.txt


# =========================================================
# APPLICATION CODE
# =========================================================

COPY app ./app


# =========================================================
# RUNTIME CONFIGURATION
# =========================================================

ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000


# =========================================================
# START APPLICATION
# =========================================================

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port", "8000"]