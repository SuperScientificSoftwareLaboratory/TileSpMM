#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>
#include "error_check.h"
#include "utils.h"
#include "omp.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuda_pipeline.h>

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