FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pandas scipy scikit-learn joblib
COPY src ./src
COPY benchmark.py .
CMD ["python", "benchmark.py"]
