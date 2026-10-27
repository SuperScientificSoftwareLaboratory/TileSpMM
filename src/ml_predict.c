#include "ml_predict.h"

#include "model.h"

#if TILESPMM_ML_FEATURE_COUNT != TILESPMM_MODEL_FEATURE_COUNT
#error "Model input layout does not match ml_predict.h"
#endif

void tilespmm_predict_fp16(const double *features, float *gamma, float *delta)
{
    predictparams_fp16(features, gamma, delta);
}

void tilespmm_predict_tf32(const double *features, float *gamma, float *delta)
{
    predictparams_tf32(features, gamma, delta);
}
