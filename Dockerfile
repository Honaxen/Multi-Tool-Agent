FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the FastAPI server
# Override CMD to run Gradio: python3 app.py
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]