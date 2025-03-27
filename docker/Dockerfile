# docker/Dockerfile
# Build stage
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    pip install --user -r requirements.txt

# Runtime stage
FROM python:3.9-slim

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Set up environment
RUN python -m spacy download en_core_web_sm && \
    chmod +x scripts/*.sh

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000
CMD ["python", "-m", "src.user_interface.web_api"]
