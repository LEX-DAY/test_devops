# Базовый образ
FROM python:3.9-slim

# Установка зависимостей
WORKDIR /app
COPY server_client/modules/client.py /app/client.py
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Разрешаем X11 для отображения окон
ENV DISPLAY=:0

# Запуск клиента
CMD ["python", "client.py"]
