# Change only this number for the target architecture: 120, 90, 80, etc.
CUDA_ARCH = 120

# Both platform preparation and normal compilation use these settings.
CC = nvcc
GCC = gcc
CUDA_INSTALL_PATH ?= $(shell dirname $$(command -v $(CC)))/..
INC = -Isrc -I$(CUDA_INSTALL_PATH)/include
# Retain the verified CUDA 12.8 / host compiler compatibility option.
NVCC_FLAGS = -U_GNU_SOURCE -O3 -w -m64 -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -Xptxas -v
OPTIONS = -Xcompiler -fopenmp -O3
CUDA_LIBS = -ldl -L$(CUDA_INSTALL_PATH)/lib64 -lcublas -lcusparse -Xlinker -rpath=$(CUDA_INSTALL_PATH)/lib64

# CHECK applies to BOTH precisions.
CHECK ?= 1
EXPERIMENT_FLAGS = -DCHECK=$(CHECK)
