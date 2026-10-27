#ifndef _UTILS_
#define _UTILS_
#include "common.h"

void binary_search_right_boundary_item_kernel(const MAT_PTR_TYPE *row_pointer,
                                              const MAT_PTR_TYPE key_input,
                                              const int size,
                                              int *colpos,
                                              MAT_PTR_TYPE *nnzpos)
{
    int start = 0;
    int stop = size - 1;
    MAT_PTR_TYPE median;
    MAT_PTR_TYPE key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    *colpos = start - 1;
    *nnzpos = key_input - row_pointer[*colpos];
}

// in-place exclusive scan
void exclusive_scan(MAT_PTR_TYPE *input, int length)
{
    if (length == 0 || length == 1)
        return;

    MAT_PTR_TYPE old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}
void exclusive_scan_char(unsigned char *input, int length)
{
    if (length == 0)
        return;
    if (length == 1)
    {
        input[0] = 0;
        return;
    }

    unsigned char old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}
void swap_key(int *a, int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

void swap_val(MAT_VAL_TYPE *a, MAT_VAL_TYPE *b)
{
    MAT_VAL_TYPE tmp = *a;
    *a = *b;
    *b = tmp;
}

// quick sort key-value pair (child function)
int partition_key_val_pair(int *key, MAT_VAL_TYPE *val, int length, int pivot_index)
{
    int i = 0;
    int small_length = pivot_index;

    int pivot = key[pivot_index];
    swap_key(&key[pivot_index], &key[pivot_index + (length - 1)]);
    swap_val(&val[pivot_index], &val[pivot_index + (length - 1)]);

    for (; i < length; i++)
    {
        if (key[pivot_index + i] < pivot)
        {
            swap_key(&key[pivot_index + i], &key[small_length]);
            swap_val(&val[pivot_index + i], &val[small_length]);
            small_length++;
        }
    }

    swap_key(&key[pivot_index + length - 1], &key[small_length]);
    swap_val(&val[pivot_index + length - 1], &val[small_length]);

    return small_length;
}

// quick sort key-value pair (main function)
void quick_sort_key_val_pair(int *key, MAT_VAL_TYPE *val, int length)
{
    if (length == 0 || length == 1)
        return;

    int small_length = partition_key_val_pair(key, val, length, 0);
    quick_sort_key_val_pair(key, val, small_length);
    quick_sort_key_val_pair(&key[small_length + 1], &val[small_length + 1], length - small_length - 1);
}

// quick sort key (child function)
int partition_key(int *key, int length, int pivot_index)
{
    int mid = length / 2;
    swap_key(&key[pivot_index], &key[pivot_index + mid]);

    int i = 0;
    int small_length = pivot_index;

    int pivot = key[pivot_index];
    swap_key(&key[pivot_index], &key[pivot_index + (length - 1)]);

    for (; i < length; i++)
    {
        if (key[pivot_index + i] < pivot)
        {
            swap_key(&key[pivot_index + i], &key[small_length]);
            small_length++;
        }
    }
    swap_key(&key[pivot_index + length - 1], &key[small_length]);
    return small_length; 
}

// quick sort key (main function)
void quick_sort_key(int *key, int length)
{
    if (length == 0 || length == 1)
        return;

    int small_length = partition_key(key, length, 0);
    quick_sort_key(key, small_length);
    quick_sort_key(&key[small_length + 1], length - small_length - 1);
}

void matrix_transposition(const int m,
                          const int n,
                          const MAT_PTR_TYPE nnz,
                          const MAT_PTR_TYPE *csrRowPtr,
                          const int *csrColIdx,
                          const MAT_VAL_TYPE *csrVal,
                          int *cscRowIdx,
                          MAT_PTR_TYPE *cscColPtr,
                          MAT_VAL_TYPE *cscVal)
{
    // histogram in column pointer
    memset(cscColPtr, 0, sizeof(MAT_PTR_TYPE) * (n + 1));
    for (MAT_PTR_TYPE i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }

    // prefix-sum scan to get the column pointer
    exclusive_scan(cscColPtr, n + 1);

    MAT_PTR_TYPE *cscColIncr = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (n + 1));
    checkHostAlloc(cscColIncr, sizeof(MAT_PTR_TYPE) * (n + 1), "cscColIncr");
    memcpy(cscColIncr, cscColPtr, sizeof(MAT_PTR_TYPE) * (n + 1));

    // insert nnz to csc
    for (int row = 0; row < m; row++)
    {
        for (MAT_PTR_TYPE j = csrRowPtr[row]; j < csrRowPtr[row + 1]; j++)
        {
            int col = csrColIdx[j];

            cscRowIdx[cscColIncr[col]] = row;
            cscVal[cscColIncr[col]] = csrVal[j];
            cscColIncr[col]++;
        }
    }
    free(cscColIncr);
}

__device__ __forceinline__
void store_transposed_mma_m16n8k16(float *reg, float *s_smem_base, int stride)
{
    int lane_id = threadIdx.x % 32;

    int group_row = lane_id >> 2;
    int group_col = (lane_id & 3) << 1;

    s_smem_base[(group_col) * stride + group_row] = reg[0];
    s_smem_base[(group_col + 1) * stride + group_row] = reg[1];

    s_smem_base[(group_col) * stride + (group_row + 8)] = reg[2];
    s_smem_base[(group_col + 1) * stride + (group_row + 8)] = reg[3];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tensor Core MMA
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
m16n8k16
FP16 x FP16 -> FP32
*/
__device__ __forceinline__
void mma_m16n8k16_f32(float *d, unsigned *a, unsigned *b, float *c)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__
void mma_m16n8k8_tf32(float *d, unsigned *a, unsigned *b, float *c)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__
void ldmatrix_m16n16_x4_trans(unsigned *reg, void *smem_ptr)
{
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg[0]), "=r"(reg[2]), "=r"(reg[1]), "=r"(reg[3])
        : "r"(smem_addr));
}

__device__ __forceinline__
void ldmatrix_m16n8_x2_trans(unsigned *reg, void *smem_ptr)
{
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];"
        : "=r"(reg[0]), "=r"(reg[1])
        : "r"(smem_addr));
}

__device__ __forceinline__
void cp_async_int4(void* smem_ptr, const void* glob_ptr)
{
    uint32_t smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(glob_ptr));
}

__device__ __forceinline__
void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__
void cp_async_wait_group()
{
    asm volatile(
        "cp.async.wait_group %0;\n"
        :: "n"(N));
}

__device__ __forceinline__
int get_smem_idx_swizzled(int row, int col)
{
    return row * STRIDE_N + (col ^ ((row & 3) << 3));
}

void get_matrix_features(int rowA, int colA, MAT_PTR_TYPE nnzA, int *csrColIdxA, MAT_PTR_TYPE *csrRowPtrA, double *features) 
{
    features[0] = 0.0; 
    features[1] = 0.0;

    features[2] = (double)rowA;
    features[3] = (double)colA;
    features[4] = (double)nnzA;
    features[5] = (double)nnzA / ((double)rowA * colA);

    int min_row_nnz = 2147483647, max_row_nnz = -1;
    
    #pragma omp parallel for reduction(min:min_row_nnz) reduction(max:max_row_nnz)
    for (int i = 0; i < rowA; i++) {
        int len = csrRowPtrA[i+1] - csrRowPtrA[i];
        if (len < min_row_nnz) min_row_nnz = len;
        if (len > max_row_nnz) max_row_nnz = len;
    }
    
    features[6] = (double)nnzA / rowA; // avg_row_nnz
    features[7] = (double)max_row_nnz;
    features[8] = (double)min_row_nnz;
}
#endif