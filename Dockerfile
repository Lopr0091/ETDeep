# Imagen base con Python 3.10 y Debian slim
FROM python:3.10-slim

# Evita prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar Rust (necesario para transformers y DVC)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copiar el archivo de dependencias (desde la raíz ahora)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto al contenedor
COPY . .

# Exponer el puerto de Gradio
EXPOSE 7860

# Comando por defecto
CMD ["python", "webapp/app.py"]
