#ifndef ML_PREDICT_H
#define ML_PREDICT_H

#ifdef __cplusplus
extern "C" {
#endif

#define TILESPMM_ML_FEATURE_COUNT 11

/* gamma/delta placeholders, rowA, colA, nnzA, sparsity, avg_row_nnz,
 * max_row_nnz, min_row_nnz, compute capability (major + minor / 10.0), logical colB.
 * Candidate selection overwrites the gamma/delta placeholders internally.
 */
void tilespmm_predict_fp16(const double *features, float *gamma, float *delta);
void tilespmm_predict_tf32(const double *features, float *gamma, float *delta);

#ifdef __cplusplus
}
#endif

#endif // ML_PREDICT_H
