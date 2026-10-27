#!/bin/sh
set -eu

if [ "$#" -gt 1 ]; then
    echo "Usage: sh prepare_prefetch.sh [gpu_id]" >&2
    exit 1
fi
device=${1:-0}
case "$device" in
    ''|*[!0-9]*) echo "gpu_id must be a nonnegative CUDA device index." >&2; exit 1 ;;
esac

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$project_dir"
if [ ! -s src/model.h ]; then
    echo "Missing model source: src/model.h" >&2
    exit 1
fi

# Each explicit preparation discards the previous generated files first.
sh pre/clean_artifacts.sh lib
# On failure, retain generated files for inspection until the next run.
trap 'exit 1' HUP INT TERM

# Serial, stop-on-error preparation, even if MAKEFLAGS requests parallel/keep-going.
make --no-print-directory -j1 -S -f pre/prepare.mk PREPARE_FROM_SCRIPT=1 DEVICE="$device"
for name in model.so prefetch_probe_fp16 prefetch_probe_tf32 \
            prefetch_fp16.h prefetch_tf32.h platform_config.mk; do
    test -s "lib/$name"
done
echo "Platform prepared for CUDA device $device. Next: make fp16, make tf32, or make all."
