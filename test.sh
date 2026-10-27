#!/bin/bash
GPU_ID=${1:-0}
COLB=(32 128)
EXECUTABLES=(./bin/tilespmm_fp16 ./bin/tilespmm_tf32)

bash prepare_prefetch.sh "$GPU_ID" || exit 1
make all || exit 1

for N in "${COLB[@]}"; do
    for EXEC in "${EXECUTABLES[@]}"; do
        for matrix in data/*.mtx; do
            echo "Running: $EXEC -d $GPU_ID \"$matrix\" $N"
            "$EXEC" -d "$GPU_ID" "$matrix" "$N" || exit 1
        done
        echo ""
    done
done
