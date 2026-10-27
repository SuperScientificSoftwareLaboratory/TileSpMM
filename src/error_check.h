#ifndef TILESPMM_ERROR_CHECK_H
#define TILESPMM_ERROR_CHECK_H

#include <stdio.h>
#include <stdlib.h>

static inline void checkHostAlloc(const void *ptr, size_t bytes, const char *msg)
{
    if (ptr == NULL && bytes > 0)
    {
        fprintf(stderr, "Host memory allocation failed: %s, requested %zu bytes\n", msg, bytes);
        exit(EXIT_FAILURE);
    }
}

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cusparse.h>

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

static inline void checkCusparse(cusparseStatus_t status, const char *msg)
{
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        if (status == CUSPARSE_STATUS_ALLOC_FAILED)
            fprintf(stderr, "cuSPARSE memory allocation failed: %s\n", msg);
        else
            fprintf(stderr, "cuSPARSE error %s: %d\n", msg, (int)status);
        exit(EXIT_FAILURE);
    }
}
#endif

#endif
