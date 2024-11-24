FROM python:3.10-slim
WORKDIR /app
COPY app.py requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Install pytesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Expose port 5000
EXPOSE 5000

# Define the command to run the app
CMD ["streamlit", "run", "app.py"]