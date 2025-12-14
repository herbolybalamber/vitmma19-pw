# Dockerfile for LegalTextDecoder deep learning project

# NVIDIA CUDA base image with Python 3.12 for GPU support
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY pyproject.toml ./

# Exclude some packages from the install
# For example scipy was only needed for EDA
# and torch is installed with the right version in the base image
RUN printf "scipy\ntorch\n" > /tmp/exclude.txt && \
    uv pip install --system --no-cache -r pyproject.toml --exclude /tmp/exclude.txt

# Copy source code and configuration
COPY src/ src/
COPY config.yaml .
COPY run.sh .

# Make run.sh executable
RUN chmod +x run.sh

# Create directories for data and output (to be mounted as volumes)
RUN mkdir -p /app/data /app/data /app/output /app/logs /app/models /app/media

# Set environment variables for better Python behavior in containers
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the entrypoint to run the training pipeline by default
# You can override this with: docker run ... python -m src.04_inference
CMD ["bash", "/app/run.sh"]
