FROM python:3.11-slim

# System dependencies:
#   tesseract-ocr        - required by pytesseract for OCR
#   libgl1, libglib2.0-0  - required by opencv-python's shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first. yolov5's normal install path pulls in the
# full CUDA build (2GB+ of GPU libraries this container will never use,
# since Render has no GPU) - installing the CPU-only wheel first means
# later installs see torch already satisfied and skip the CUDA version.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Directories the app writes to at runtime (kept out of the image via
# .dockerignore, but the app expects them to exist).
RUN mkdir -p database output/snapshots

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]