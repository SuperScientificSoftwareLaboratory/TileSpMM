#ifndef TILESPMM_COMMON_H
#define TILESPMM_COMMON_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>
#include <omp.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuda_pipeline.h>
#include <dlfcn.h>

#ifndef MAT_VAL_TYPE
#define MAT_VAL_TYPE half
#endif

#ifndef MAT_ANS_TYPE
#define MAT_ANS_TYPE float
#endif

#ifndef BENCH_REPEAT
#define BENCH_REPEAT 1000
#endif

#ifndef WARMUP_NUM
#define WARMUP_NUM 100
#endif

#ifndef MAT_PTR_TYPE
#define MAT_PTR_TYPE int
#endif

#ifndef STRIDE_N
#define STRIDE_N 64
#endif

#ifndef Tile_N
#define Tile_N 32
#endif

#ifndef num_f
#define num_f 240
#endif

#ifndef num_b
#define num_b 15
#endif

#ifndef PADDING_HALF
#define PADDING_HALF 8
#endif

#ifndef PADDING_FLOAT
#define PADDING_FLOAT 8
#endif

#ifndef grid_y_limit
#define grid_y_limit 200
#endif

#ifndef stride_padded_half
#define stride_padded_half (STRIDE_N + PADDING_HALF)
#endif

#ifndef stride_padded_float
#define stride_padded_float (STRIDE_N + PADDING_FLOAT)
#endif

// 函数实现
static inline void checkHostAlloc(const void *ptr, size_t bytes, const char *msg)
{
    if (ptr == NULL && bytes > 0)
    {
        fprintf(stderr, "Host memory allocation failed: %s, requested %zu bytes\n", msg, bytes);
        exit(EXIT_FAILURE);
    }
}

static inline void checkCuda(cudaError_t e, const char *msg)
{
    if (e != cudaSuccess)
    {
        if (e == cudaErrorMemoryAllocation)
            fprintf(stderr, "GPU out of memory: %s\n", msg);
        else
            fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

static unsigned int get_gpu_clock()
{
    typedef int (*Init)(void);
    typedef int (*Find)(const char *, void **);
    typedef int (*Clock)(void *, unsigned int, unsigned int *);
    static void *device = NULL;
    static Clock clock = NULL;
    static bool initialized = false;
    if (!initialized)
    {
        initialized = true;
        void *library = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        Init init = library ? (Init)dlsym(library, "nvmlInit_v2") : NULL;
        Find find = library ? (Find)dlsym(library, "nvmlDeviceGetHandleByPciBusId_v2") : NULL;
        clock = library ? (Clock)dlsym(library, "nvmlDeviceGetClockInfo") : NULL;
        int id = 0;
        char pci[32];
        checkCuda(cudaGetDevice(&id), "cudaGetDevice");
        checkCuda(cudaDeviceGetPCIBusId(pci, sizeof(pci), id), "cudaDeviceGetPCIBusId");
        if (!init || !find || !clock || init() || find(pci, &device)) clock = NULL;
    }
    unsigned int mhz = 0;
    return clock && clock(device, 1, &mhz) == 0 ? mhz : 0;
}

#endif
