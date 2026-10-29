#ifndef TILESPMM_MODEL_FEATURES_H
#define TILESPMM_MODEL_FEATURES_H

#include <limits.h>
#include <math.h>
#include "ml_predict.h"

// 使用对称展开后的原始 CSR 统计；空行参与最小值计算，colB 为实际列宽
static inline int features_from_stats(int rows, int cols, long long nnz, int max_row_nnz, int min_row_nnz,
    double capability, int colB, double *features)
{
    if (!features || rows <= 0 || cols <= 0 || nnz < 0 ||
        min_row_nnz < 0 || max_row_nnz < min_row_nnz ||
        !isfinite(capability) || capability <= 0.0 || colB <= 0)
        return 0;

    features[0] = 0.0;
    features[1] = 0.0;
    features[2] = (double)rows;
    features[3] = (double)cols;
    features[4] = (double)nnz;
    features[5] = 1.0 - (double)nnz / ((double)rows * (double)cols);
    features[6] = (double)nnz / (double)rows;
    features[7] = (double)max_row_nnz;
    features[8] = (double)min_row_nnz;
    features[9] = capability;
    features[10] = (double)colB;
    return 1;
}

template <typename Pointer>
static inline int features_from_csr(int rows, int cols, long long nnz, const Pointer *row_ptr,
    double capability, int colB, double *features)
{
    if (!row_ptr || rows <= 0 || row_ptr[0] != 0 ||
        (long long)row_ptr[rows] != nnz)
        return 0;

    int min_nnz = INT_MAX;
    int max_nnz = 0;
    int invalid = 0;
    #pragma omp parallel for reduction(min:min_nnz) reduction(max:max_nnz) reduction(|:invalid)
    for (int row = 0; row < rows; ++row) {
        long long count = (long long)row_ptr[row + 1] - (long long)row_ptr[row];
        if (count < 0 || count > INT_MAX) {
            invalid = 1;
        } else {
            if (count < min_nnz) min_nnz = (int)count;
            if (count > max_nnz) max_nnz = (int)count;
        }
    }
    if (invalid)
        return 0;

    return features_from_stats(
        rows, cols, nnz, max_nnz, min_nnz, capability, colB, features);
}

#endif
