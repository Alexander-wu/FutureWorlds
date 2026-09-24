FROM python:3.13-slim
ENV USE_TF=0 USE_FLAX=0 PYTHONUNBUFFERED=1
WORKDIR /opt/futureworlds
COPY requirements-linux.txt .
RUN python -m pip install --no-cache-dir -r requirements-linux.txt
COPY . .
RUN python -m pip install --no-cache-dir --no-deps -e .
ENTRYPOINT ["python", "-m", "futureworlds"]
