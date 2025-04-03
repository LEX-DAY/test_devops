# Базовый образ NVIDIA с поддержкой TensorRT
FROM nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Установка дополнительных Python-зависимостей
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . /app

# Команда для запуска инференса при старте контейнера
CMD ["python3", "main.py"]