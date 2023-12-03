# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables
ENV MAX_LENGTH=30
ENV MODEL_NAME="facebook/rag-token-nq"

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Install any needed packages specified in requirements.txt including pytorch
RUN pip install --no-cache-dir transformers fastapi uvicorn pydantic datasets faiss-cpu pytorch

# Run app.py when the container launches
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5000","--log-level", "info"]