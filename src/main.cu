#include "common.h"
#include "mmio_highlevel.h"
#include "utils.h"
#include "csr2tile.h"
#include "tilespmm_cuda.h"
#include <cublas_v2.h>
#include "hetero_split.h"
#include "model_features.h"
#include <cmath>
#include <limits.h>

#ifndef CHECK
#define CHECK 1
#endif

void cusparse_spmm_all(int m, int k, int nnzA, 
                        MAT_PTR_TYPE *h_csrRowPtrA, int *h_csrColIdxA, MAT_VAL_TYPE *h_csrValA, 
                        int n, int ldb, MAT_VAL_TYPE *h_B, MAT_ANS_TYPE *h_C, FILE *fout)
{
    struct timeval t1, t2;

    int *d_csrRowPtrA, *d_csrColIdxA;
    MAT_VAL_TYPE *d_csrValA, *d_B;
    MAT_ANS_TYPE *d_C;

    size_t sizeRowPtr = (size_t)(m + 1) * sizeof(MAT_PTR_TYPE);
    size_t sizeColIdx = (size_t)nnzA * sizeof(int);
    size_t sizeValA   = (size_t)nnzA * sizeof(MAT_VAL_TYPE); 
    size_t sizeB      = (size_t)k * (size_t)ldb * sizeof(MAT_VAL_TYPE);
    
    size_t sizeC_padded = (size_t)m * (size_t)n * sizeof(MAT_ANS_TYPE);

    checkCuda(cudaMalloc((void**)&d_csrRowPtrA, sizeRowPtr), "malloc d_csrRowPtrA");
    checkCuda(cudaMalloc((void**)&d_csrColIdxA, sizeColIdx), "malloc d_csrColIdxA");
    checkCuda(cudaMalloc((void**)&d_csrValA, sizeValA), "malloc d_csrValA");
    checkCuda(cudaMalloc((void**)&d_B, sizeB), "malloc d_B");
    checkCuda(cudaMalloc((void**)&d_C, sizeC_padded), "malloc d_C padded");

    checkCuda(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeRowPtr, cudaMemcpyHostToDevice), "memcpy rowPtr");
    checkCuda(cudaMemcpy(d_csrColIdxA, h_csrColIdxA, sizeColIdx, cudaMemcpyHostToDevice), "memcpy colIdx");
    checkCuda(cudaMemcpy(d_csrValA, h_csrValA, sizeValA, cudaMemcpyHostToDevice), "memcpy val");
    checkCuda(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice), "memcpy B");

    checkCuda(cudaMemset(d_C, 0, sizeC_padded), "memset C");

    gettimeofday(&t1, NULL);

    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseSetStream(handle, cudaStreamLegacy);

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
#ifdef TFLOAT
    cusparseCreateCsr(&matA, m, k, nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseCreateDnMat(&matB, k, n, ldb, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
#else
    cusparseCreateCsr(&matA, m, k, nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);

    cusparseCreateDnMat(&matB, k, n, ldb, d_B, CUDA_R_16F, CUSPARSE_ORDER_ROW);

#endif
    cusparseCreateDnMat(&matC, m, n, n, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW);


    float alpha = 1.0f;
    float beta  = 0.0f;
    
    cudaDataType computeType = CUDA_R_32F; 

    size_t bufferSize = 0;
    void *dBuffer = NULL;
    
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    
    checkCuda(cudaMalloc(&dBuffer, bufferSize), "malloc buffer");
    gettimeofday(&t2, NULL);
    double pre_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    // Continue the same CSV row after TileSpMM's three timing fields.
    fprintf(fout, "%.9g,", pre_time);
    fflush(fout);

    for(int i=0; i<WARMUP_NUM; i++) {
        cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
    }
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate");

    checkCuda(cudaEventRecord(start, cudaStreamLegacy), "cudaEventRecord");

    for(int i=0; i<BENCH_REPEAT; i++) {
        cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
    }

    checkCuda(cudaEventRecord(stop, cudaStreamLegacy), "cudaEventRecord");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");

    float milliseconds_cusparse = 0;
    checkCuda(cudaEventElapsedTime(&milliseconds_cusparse, start, stop), "cudaEventElapsedTime");
    double time_ms = milliseconds_cusparse / BENCH_REPEAT;
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy");


    double total_flops = 2.0 * (double)nnzA * (double)n;
    double tflops = (total_flops / (time_ms / 1000.0)) / 1e12;

    printf("  cuSPARSE runtime %4.3f ms, %4.3f Tflops\n", time_ms, tflops);

    fprintf(fout, "%.9g,%.9g\n", time_ms, tflops);
    fflush(fout);

    // beta is zero, so the last cuSPARSE iteration contains one SpMM result.
    // This copy is outside both preprocessing and runtime measurements.
    if (h_C != NULL)
        checkCuda(cudaMemcpy(h_C, d_C, sizeC_padded, cudaMemcpyDeviceToHost), "copy cuSPARSE result");

    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroySpMat(matA);
    cusparseDestroy(handle);
    checkCuda(cudaFree(dBuffer), "free buffer");
    checkCuda(cudaFree(d_csrRowPtrA), "free d_A_ptr");
    checkCuda(cudaFree(d_csrColIdxA), "free d_A_idx");
    checkCuda(cudaFree(d_csrValA), "free d_A_val");
    checkCuda(cudaFree(d_B), "free d_B");
    checkCuda(cudaFree(d_C), "free d_C");

}

int main(int argc, char **argv)
{
    if (argc < 5 || strcmp(argv[1], "-d") != 0)
    {
        printf("Usage: %s -d device_id matrix.mtx colB\n", argv[0]);
        return EXIT_FAILURE;
    }
    printf("--------------------------------!!!!!!!!------------------------------------\n");

    struct timeval t1, t2;
    int rowA;
    int colA;
    MAT_PTR_TYPE nnzA;
    int isSymmetricA;
    MAT_VAL_TYPE *csrValA;
    int *csrColIdxA;
    MAT_PTR_TYPE *csrRowPtrA;

    int device_id = atoi(argv[2]);
    printf("device_id = %i\n", device_id);

    char *filename;
    filename = argv[3];
    printf("MAT: -------------- %s --------------\n", filename);
    gettimeofday(&t1, NULL);
    mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &csrRowPtrA, &csrColIdxA, &csrValA, filename);
    gettimeofday(&t2, NULL);

    double time_loadmat = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("  input matrix A: ( %i, %i ) nnz = %i\n  loadfile time    = %4.5f sec\n", rowA, colA, nnzA, time_loadmat / 1000.0);
    int rowB = colA;
    int colB = atoi(argv[4]);
    if (colB <= 0 || colB > INT_MAX - 63)
    {
        fprintf(stderr, "colB must be positive and fit the padded kernel width.\n");
        return EXIT_FAILURE;
    }
    int colB_padded = colB <= 32 ? 32 : ((colB + 63) / 64) * 64;
    printf("  colB = %d, padded colB = %d\n", colB, colB_padded);

    // Fixed seed makes the random integer inputs reproducible on this platform.
    srand(1325);
    
    for (size_t i = 0; i < nnzA; i++)
    {
        float val = (float)(1 + rand() % 3);

#ifdef TFLOAT
        csrValA[i] = val; 
#else
        csrValA[i] = __float2half(val);
#endif
    }

    size_t matrixB_elements = (size_t)rowB * (size_t)colB_padded;
    MAT_VAL_TYPE *matrixB = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * matrixB_elements);
    checkHostAlloc(matrixB, sizeof(MAT_VAL_TYPE) * matrixB_elements, "matrixB");
    
    for (int row = 0; row < rowB; row++)
    {
        for (int col = 0; col < colB_padded; col++)
        {
            size_t i = (size_t)row * (size_t)colB_padded + (size_t)col;
            float val = col < colB ? (float)(1 + rand() % 3) : 0.0f;
        
#ifdef TFLOAT
            matrixB[i] = val;
#else       
            matrixB[i] = __float2half(val);
#endif
        }
    }

    
    checkCuda(cudaSetDevice(device_id), "select device");
    cudaDeviceProp deviceProp;
    checkCuda(cudaGetDeviceProperties(&deviceProp, device_id), "get device properties");
    if (!prefetch_device_matches(deviceProp))
    {
        fprintf(stderr, "Prefetch resources were prepared for a different GPU. "
                        "Set CUDA_ARCH in build_config.mk, run sh prepare_prefetch.sh %d, "
                        "then run make all.\n", device_id);
        return EXIT_FAILURE;
    }
    int clockRateKHz = 0;
    checkCuda(cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, device_id), "cudaDeviceGetAttribute");
    char* dev_name = deviceProp.name;
    printf("---------------------------------------------------------------------------------------------\n");
    printf("  Device [ %i ] %s @ %4.2f MHz\n", device_id, deviceProp.name, clockRateKHz * 1e-3f);

    gettimeofday(&t1, NULL);
    double ml_features[TILESPMM_ML_FEATURE_COUNT];
    double capability = deviceProp.major + deviceProp.minor / 10.0;
    if (!features_from_csr(
            rowA, colA, nnzA, csrRowPtrA, capability, colB, ml_features))
    {
        fprintf(stderr, "Cannot extract model features from the input CSR matrix.\n");
        return EXIT_FAILURE;
    }
    
    float best_gamma = 0.0f;
    float best_delta = 0.0f;
#ifdef TFLOAT
    tilespmm_predict_tf32(ml_features, &best_gamma, &best_delta);
#else
    tilespmm_predict_fp16(ml_features, &best_gamma, &best_delta);
#endif

    int rowA_64 = 0, rowA_16 = 0, rowA_8 = 0;
    MAT_PTR_TYPE nnzA_64 = 0, nnzA_16 = 0, nnzA_8 = 0;
    MAT_PTR_TYPE *ptr_64=NULL, *ptr_16=NULL, *ptr_8=NULL;
    int *idx_64=NULL, *idx_16=NULL, *idx_8=NULL;
    MAT_VAL_TYPE *val_64=NULL, *val_16=NULL, *val_8=NULL;
    int *orig_64=NULL, *orig_16=NULL, *orig_8=NULL;

    matrix_split(rowA, colA, nnzA, csrRowPtrA, csrColIdxA, csrValA, best_gamma, best_delta,
                &rowA_64, &nnzA_64, &ptr_64, &idx_64, &val_64, &orig_64,
                &rowA_16, &nnzA_16, &ptr_16, &idx_16, &val_16, &orig_16,
                &rowA_8,  &nnzA_8,  &ptr_8,  &idx_8,  &val_8,  &orig_8);
            
    Tile_matrix *mat64 = (Tile_matrix *)malloc(sizeof(Tile_matrix));
    checkHostAlloc(mat64, sizeof(Tile_matrix), "mat64");
    Tile_matrix *mat16 = (Tile_matrix *)malloc(sizeof(Tile_matrix));
    checkHostAlloc(mat16, sizeof(Tile_matrix), "mat16");
    Tile_matrix *mat8  = (Tile_matrix *)malloc(sizeof(Tile_matrix));
    checkHostAlloc(mat8, sizeof(Tile_matrix), "mat8");
    memset(mat64, 0, sizeof(Tile_matrix));
    memset(mat16, 0, sizeof(Tile_matrix));
    memset(mat8,  0, sizeof(Tile_matrix));

    Tile_workspace tile_workspace = {};
#ifdef TFLOAT
    if(rowA_64 > 0) Tile_create(mat64, rowA_64, colA, nnzA_64, colB, ptr_64, idx_64, val_64, orig_64, 64, 16, &tile_workspace);
    if(rowA_16 > 0) Tile_create(mat16, rowA_16, colA, nnzA_16, colB, ptr_16, idx_16, val_16, orig_16, 16, 8, &tile_workspace);
    if(rowA_8 > 0)  Tile_create(mat8,  rowA_8,  colA, nnzA_8,  colB, ptr_8,  idx_8,  val_8, orig_8, 8,  8, &tile_workspace);
#else
    if(rowA_64 > 0) Tile_create(mat64, rowA_64, colA, nnzA_64, colB, ptr_64, idx_64, val_64, orig_64, 64, 32, &tile_workspace);
    if(rowA_16 > 0) Tile_create(mat16, rowA_16, colA, nnzA_16, colB, ptr_16, idx_16, val_16, orig_16, 16, 16, &tile_workspace);
    if(rowA_8 > 0)  Tile_create(mat8,  rowA_8,  colA, nnzA_8,  colB, ptr_8,  idx_8,  val_8, orig_8,  8,  16, &tile_workspace);
#endif
    free_tile_workspace(&tile_workspace);
    gettimeofday(&t2, NULL);
    double pre_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    printf("  CPU preprocessing time = %4.5f s\n", pre_time / 1000.0);

    size_t total_elements = (size_t)rowA * (size_t)colB;
    size_t padded_elements = (size_t)rowA * (size_t)colB_padded;
    MAT_ANS_TYPE *matrixC = NULL;
    MAT_ANS_TYPE *matrixC_cusparse = NULL;
#if CHECK
    matrixC = (MAT_ANS_TYPE *)malloc(sizeof(MAT_ANS_TYPE) * padded_elements);
    checkHostAlloc(matrixC, sizeof(MAT_ANS_TYPE) * padded_elements, "matrixC");
    matrixC_cusparse = (MAT_ANS_TYPE *)malloc(sizeof(MAT_ANS_TYPE) * total_elements);
    checkHostAlloc(matrixC_cusparse, sizeof(MAT_ANS_TYPE) * total_elements, "matrixC_cusparse");
#endif

#ifdef TFLOAT
    const char *csv_path = "TileSpMM-TF32.csv";
#else
    const char *csv_path = "TileSpMM-FP16.csv";
#endif
    FILE *fout = fopen(csv_path, "a");
    if (fout == NULL)
    {
        printf("Writing results fails.\n");
        return EXIT_FAILURE;
    }

    // Start the row with all matrix and partition information already available.
    fprintf(fout, "%s,%d,%d,%d,%d,%d,%d,%d,%.1f,%.1f,", filename, rowA, colA, nnzA, colB, 
        mat64->tilenum, mat16->tilenum, mat8->tilenum, best_gamma, best_delta);
    fflush(fout);

    call_tilespmm_cuda(mat64, rowA_64, nnzA_64, mat16, rowA_16, nnzA_16, mat8, rowA_8, nnzA_8, 
                        colA, colB_padded, nnzA, matrixB, matrixC, deviceProp, pre_time, colB, fout);

    cusparse_spmm_all(rowA, colA, nnzA, csrRowPtrA, csrColIdxA, csrValA, colB, colB_padded, matrixB, matrixC_cusparse, fout);

#if CHECK
    size_t error_count_cuda = 0;
    for (int row = 0; row < rowA; row++)
    {
        for (int col = 0; col < colB; col++)
        {
            double tile_value = (double)matrixC[(size_t)row * (size_t)colB_padded + (size_t)col];
            double cusparse_value = (double)matrixC_cusparse[(size_t)row * (size_t)colB + (size_t)col];
            if (!std::isfinite(tile_value) || !std::isfinite(cusparse_value) || fabs(cusparse_value - tile_value) > 1e-3)
            {
                if (error_count_cuda < 10)
                    printf("  c[%d][%d], tilespmm: %lf, cusparse: %lf\n", row, col, tile_value, cusparse_value);
                error_count_cuda++;
            }
        }
    }
    if (error_count_cuda == 0)
        printf("  Check... PASS!\n");
    else
        printf("  Check... NO PASS! error_count_cuda = %zu\n", error_count_cuda);
#endif
    fclose(fout);
    free(matrixC);
    free(matrixC_cusparse);

    Tile_destroy(mat64); Tile_destroy(mat16); Tile_destroy(mat8);
    free(mat64->orig_tile_row_idx); free(mat16->orig_tile_row_idx); free(mat8->orig_tile_row_idx);
    free(mat64); free(mat16); free(mat8);
    if(ptr_64) free(ptr_64); if(idx_64) free(idx_64); if(val_64) free(val_64); if(orig_64) free(orig_64);
    if(ptr_16) free(ptr_16); if(idx_16) free(idx_16); if(val_16) free(val_16); if(orig_16) free(orig_16);
    if(ptr_8)  free(ptr_8);  if(idx_8)  free(idx_8);  if(val_8)  free(val_8);  if(orig_8)  free(orig_8);

    free(csrValA);
    free(csrColIdxA);
    free(csrRowPtrA);
    free(matrixB);
    return EXIT_SUCCESS;
}
