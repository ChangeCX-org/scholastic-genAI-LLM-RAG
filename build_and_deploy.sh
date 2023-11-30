#!/bin/bash

# Variables
project_id="changecx-356309"
location="us-central1"
image_name="rag-tokenizer-container"

# Set the project ID
gcloud config set project $project_id

# Authenticate (uncomment this if you're running this script in a local environment)
# gcloud auth login

# Build the Docker image
docker build -t $image_name .

# Tag the image for Google Container Registry
docker tag $image_name us.gcr.io/$project_id/$image_name

# Push the image to Google Container Registry
gcloud auth configure-docker us.gcr.io
docker push us.gcr.io/$project_id/$image_name

echo "Image $image_name pushed to Google Container Registry"