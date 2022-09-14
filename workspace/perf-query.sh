#!/bin/bash
perf_analyzer \
        -m $MODEL \
        -a \
        -i grpc \
        -u $TRITON_ENDPOINT:8001 \
        --percentile 95 \
        --max-threads $MAX_THREADS \
        --request-distribution constant \
        --measurement-interval 300000 \
        --concurrency-range $MIN_CONCURRENCY:$MAX_CONCURRENCY:$STEP_CONCURRENCY \
        --input-data /workspace/data.json \
        -b $BATCH_SIZE