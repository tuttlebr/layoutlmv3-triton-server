#!/bin/bash

clear
docker compose -f docker-compose-tesseract.yaml down --remove-orphans
docker compose down --remove-orphans
docker system prune -f

docker compose -f docker-compose-tesseract.yaml build
docker compose build

docker compose -f docker-compose-tesseract.yaml up --remove-orphans

docker compose up layoutlmv3-triton-server -d --remove-orphans
TRITON_HEALTH=000
until [ $TRITON_HEALTH = 200 ]
    do 
        echo "Waiting for Triton to load..."
        TRITON_HEALTH=$(curl -m 1 -L -s -o /dev/null -w %{http_code} localhost:8000/v2/health/live)
        sleep 5
    done

curl -s localhost:8000/v2 | jq

docker compose up layoutlmv3-triton-client --remove-orphans

docker compose up layoutlmv3-triton-client-query --remove-orphans

docker compose -f docker-compose-tesseract.yaml down --remove-orphans
docker compose down --remove-orphans
