# Internal build rules used only by prepare_prefetch.sh.
include build_config.mk

ifneq ($(PREPARE_FROM_SCRIPT),1)
$(error Run sh prepare_prefetch.sh [gpu_id])
endif
OUT_DIR = lib
DEVICE ?= 0
PROBE_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -Xlinker -rpath=$(CUDA_INSTALL_PATH)/lib64

.NOTPARALLEL:
.PHONY: all
all: $(OUT_DIR)/model.so $(OUT_DIR)/prefetch_fp16.h $(OUT_DIR)/prefetch_tf32.h $(OUT_DIR)/platform_config.mk

$(OUT_DIR)/model.so: src/ml_predict.c src/ml_predict.h src/model.h
	$(GCC) -std=c11 -shared -fPIC -O3 -Wl,-soname,model.so src/ml_predict.c -o $@

$(OUT_DIR)/prefetch_probe_fp16: pre/prefetch_probe.cu src/tilespmm_cuda.h src/common.h src/utils.h build_config.mk
	$(CC) $(INC) $(NVCC_FLAGS) pre/prefetch_probe.cu -o $@ $(OPTIONS) -D MAT_VAL_TYPE=half $(PROBE_LIBS) -lineinfo

$(OUT_DIR)/prefetch_probe_tf32: pre/prefetch_probe.cu src/tilespmm_cuda.h src/common.h src/utils.h build_config.mk
	$(CC) $(INC) $(NVCC_FLAGS) pre/prefetch_probe.cu -o $@ $(OPTIONS) -D MAT_VAL_TYPE=float $(PROBE_LIBS) -lineinfo -DTFLOAT

$(OUT_DIR)/prefetch_fp16.h: $(OUT_DIR)/prefetch_probe_fp16
	./$< $(DEVICE) > $@

$(OUT_DIR)/prefetch_tf32.h: $(OUT_DIR)/prefetch_probe_tf32
	./$< $(DEVICE) > $@

$(OUT_DIR)/platform_config.mk: $(OUT_DIR)/model.so $(OUT_DIR)/prefetch_fp16.h $(OUT_DIR)/prefetch_tf32.h
	@printf 'PREPARED_CUDA_ARCH := %s\nPREPARED_DEVICE := %s\n' '$(CUDA_ARCH)' '$(DEVICE)' > $@
