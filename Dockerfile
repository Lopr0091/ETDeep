# Imagen base oficial PyTorch con CUDA
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Sistema: solo lo necesario
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instala dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt

# Copia el resto del proyecto
COPY . .

# Exponer puerto Gradio
EXPOSE 7860

# Comando para producción
CMD ["python", "webapp/app.py"]
