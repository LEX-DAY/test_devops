# Базовый образ
FROM python:3.9-slim

# Установка зависимостей
WORKDIR /app
COPY server_client/modules/server.py /app/server.py
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Запуск сервера
CMD ["python", "server.py"]
