# Imagen base oficial de Python mínima
FROM python:3.10-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Instalar solo las dependencias necesarias del sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero y luego instalar
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
