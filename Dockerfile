FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    pkg-config \
    libboost-all-dev \
    ocl-icd-opencl-dev \
    wget curl git \
    && rm -rf /var/lib/apt/lists/*

# Виртуальное окружение
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Установка зависимостей
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# PyCUDA
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:$LD_LIBRARY_PATH
RUN pip install --no-cache-dir pycuda==2024.1.1

# PyQt5
RUN pip install --no-cache-dir pyqt5==5.15.9

# Проверка
RUN python -c "import pycuda.driver as cuda; cuda.init(); print('✔ PyCUDA initialized')"

# Копирование проекта
COPY model /app/model
COPY modules /app/modules
COPY main.py /app/main.py
COPY test_img /app/test_img

WORKDIR /app
ENV PYTHONPATH="/app"
ENV DISPLAY=:0

CMD ["python", "main.py"]
