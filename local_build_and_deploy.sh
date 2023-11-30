#!/bin/bash

image_name="rag-tokenizer-container"


# Build the Docker image
docker build -t $image_name .

#Tag the image with $image_name and latest tag
docker tag $image_name:latest $image_name:latest

#start the container in background on port 80 (http)
docker run -d -p 5000:5000 $image_name:latest

#docker run -p 5000:5000 rag-tokenizer-container:latest