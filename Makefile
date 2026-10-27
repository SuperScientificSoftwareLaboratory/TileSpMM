include build_config.mk
-include lib/platform_config.mk

BIN_DIR = bin
LIB_DIR = lib
MODEL_SO = $(LIB_DIR)/model.so
ML_LIBS = -L./$(LIB_DIR) -l:model.so -Xlinker -rpath -Xlinker '$$ORIGIN/../$(LIB_DIR)'
LIBS = $(CUDA_LIBS) $(ML_LIBS)

.PHONY: all fp16 tf32 nfp16 ntf32 check-platform clean
all: fp16 tf32

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Read-only checks: make never compiles the model or probes the GPU.
check-platform:
	@test -s $(MODEL_SO) && test -s $(LIB_DIR)/prefetch_fp16.h && test -s $(LIB_DIR)/prefetch_tf32.h && test -s $(LIB_DIR)/platform_config.mk || { echo "Platform is not prepared. Run: sh prepare_prefetch.sh [gpu_id]" >&2; exit 1; }
	@test "$(PREPARED_CUDA_ARCH)" = "$(CUDA_ARCH)" || { echo "CUDA_ARCH changed. Run: sh prepare_prefetch.sh [gpu_id]" >&2; exit 1; }

fp16: check-platform | $(BIN_DIR)
	$(CC) $(INC) $(NVCC_FLAGS) $(EXPERIMENT_FLAGS) src/main.cu -o $(BIN_DIR)/tilespmm_fp16 $(LIBS) $(OPTIONS) -D MAT_VAL_TYPE=half -lineinfo

tf32: check-platform | $(BIN_DIR)
	$(CC) $(INC) $(NVCC_FLAGS) $(EXPERIMENT_FLAGS) src/main.cu -o $(BIN_DIR)/tilespmm_tf32 $(LIBS) $(OPTIONS) -D MAT_VAL_TYPE=float -lineinfo -DTFLOAT

# Prepared model and resource headers remain frozen until the script is rerun.
clean:
	sh pre/clean_artifacts.sh bin
