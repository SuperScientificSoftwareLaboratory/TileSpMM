#!/bin/sh
set -eu

# Only these two generated directories may be emptied.
case "${1:-}" in
    lib|bin) directory=$1 ;;
    *) echo "Usage: sh pre/clean_artifacts.sh lib|bin" >&2; exit 1 ;;
esac
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
artifact_dir="$project_dir/$directory"
if [ -L "$artifact_dir" ]; then
    echo "Refusing to clean a symlink: $artifact_dir" >&2
    exit 1
fi
mkdir -p "$artifact_dir"
resolved_dir=$(CDPATH= cd -- "$artifact_dir" && pwd -P)
if [ "$resolved_dir" != "$artifact_dir" ]; then
    echo "Refusing to clean a redirected directory: $artifact_dir" >&2
    exit 1
fi
if [ "$directory" = lib ] && [ -e "$artifact_dir/model.h" ]; then
    echo "Move model.h to src/model.h before cleaning lib." >&2
    exit 1
fi
# Includes hidden files and subdirectories; does not follow symlinks inside it.
find "$artifact_dir" -mindepth 1 -delete
