FROM python:3.11-slim

# Install OS packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-eng poppler-utils libmagic1 && \
    rm -rf /var/lib/apt/lists/*

# Set tesseract data path
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Start FastAPI
CMD ["uvicorn", "rag:app", "--host", "0.0.0.0", "--port", "$PORT"]
