#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// 统计行区间内的不同列数，只清理本次访问过的标记
int count_cols(int start, int end, MAT_PTR_TYPE *csrRowPtrA, int *csrColIdxA, char *flag, int *touched) 
{
    int unique_count = 0;
    int touched_count = 0;

    MAT_PTR_TYPE start_idx = csrRowPtrA[start];
    MAT_PTR_TYPE end_idx = csrRowPtrA[end];
    
    for (MAT_PTR_TYPE j = start_idx; j < end_idx; ++j) 
    {
        int c = csrColIdxA[j];
        if (flag[c] == 0) 
        {
            flag[c] = 1;
            touched[touched_count++] = c;
            unique_count++;
        }
    }
    
    for (int i = 0; i < touched_count; ++i) 
        flag[touched[i]] = 0;
        
    return unique_count;
}
 
// 对齐的8/16行窗口可能被重复查询，仅缓存精确计数，不改变阈值比较
int count_cols_cached(int start, int end, MAT_PTR_TYPE *csrRowPtrA, int *csrColIdxA,
                      char *flag, int *touched, int *count_cache, int cache_size)
{
    int length = end - start;
    if (count_cache && (start & 7) == 0)
    {
        if (length == 64)
        {
            // 一次扫描同时得到64行及其8/16行子窗口的精确去重数
            int count8[8] = {0};
            int count16[4] = {0};
            int unique_count = 0;
            int touched_count = 0;
            unsigned char *window_flag = (unsigned char *)flag;
            for (int row = start; row < end; row++)
            {
                int group8 = (row - start) / 8;
                unsigned int bit = 1u << group8;
                unsigned int pair = 3u << ((group8 / 2) * 2);
                for (MAT_PTR_TYPE j = csrRowPtrA[row]; j < csrRowPtrA[row + 1]; j++)
                {
                    int col = csrColIdxA[j];
                    unsigned int bits = window_flag[col];
                    if (bits == 0)
                    {
                        touched[touched_count++] = col;
                        unique_count++;
                    }
                    if ((bits & bit) == 0) count8[group8]++;
                    if ((bits & pair) == 0) count16[group8 / 2]++;
                    window_flag[col] = (unsigned char)(bits | bit);
                }
            }
            for (int j = 0; j < touched_count; j++) flag[touched[j]] = 0;
            int slot = start / 8;
            for (int j = 0; j < 8; j++) count_cache[slot + j] = count8[j];
            for (int j = 0; j < 4; j++) count_cache[cache_size + slot + j * 2] = count16[j];
            return unique_count;
        }
        if (length == 8 || length == 16)
        {
            int slot = start / 8 + (length == 16 ? cache_size : 0);
            if (count_cache[slot] < 0)
                count_cache[slot] = count_cols(start, end, csrRowPtrA, csrColIdxA, flag, touched);
            return count_cache[slot];
        }
    }
    return count_cols(start, end, csrRowPtrA, csrColIdxA, flag, touched);
}

// 按分组中的原始行顺序生成子 CSR，并保留行号映射
void build_sub_csr(int *group, int group_size, MAT_PTR_TYPE *csrRowPtrA, int *csrColIdxA, MAT_VAL_TYPE *csrValA,
                int *rowA_sub, MAT_PTR_TYPE *nnzA_sub, MAT_PTR_TYPE **ptr_sub, int **idx_sub, MAT_VAL_TYPE **val_sub, int **orig_row_sub) 
{
    *rowA_sub = group_size;
    if (group_size == 0) 
    {
        *ptr_sub = NULL; 
        *idx_sub = NULL; 
        *val_sub = NULL; 
        *orig_row_sub = NULL;
        *nnzA_sub = 0;
        return;
    }

    *ptr_sub = (MAT_PTR_TYPE*)malloc((group_size + 1) * sizeof(MAT_PTR_TYPE));
    checkHostAlloc(*ptr_sub, (group_size + 1) * sizeof(MAT_PTR_TYPE), "ptr_sub");
    *orig_row_sub = (int*)malloc(group_size * sizeof(int));
    checkHostAlloc(*orig_row_sub, group_size * sizeof(int), "orig_row_sub");
    (*ptr_sub)[0] = 0;

    MAT_PTR_TYPE total_sub_nnz = 0;
    for (int r = 0; r < group_size; ++r) 
    {
        int orig_r = group[r];
        (*orig_row_sub)[r] = orig_r;
        total_sub_nnz += (csrRowPtrA[orig_r + 1] - csrRowPtrA[orig_r]);
        (*ptr_sub)[r + 1] = total_sub_nnz;
    }
    *nnzA_sub = total_sub_nnz;

    if (total_sub_nnz > 0) 
    {
        *idx_sub = (int*)malloc(total_sub_nnz * sizeof(int));
        checkHostAlloc(*idx_sub, total_sub_nnz * sizeof(int), "idx_sub");
        *val_sub = (MAT_VAL_TYPE*)malloc(total_sub_nnz * sizeof(MAT_VAL_TYPE));
        checkHostAlloc(*val_sub, total_sub_nnz * sizeof(MAT_VAL_TYPE), "val_sub");
 
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < group_size; ++r) 
        {
            int orig_r = group[r];
            MAT_PTR_TYPE row_nnz = csrRowPtrA[orig_r + 1] - csrRowPtrA[orig_r];
            if (row_nnz > 0) 
            {
                MAT_PTR_TYPE offset = (*ptr_sub)[r]; 
                memcpy((*idx_sub) + offset, &csrColIdxA[csrRowPtrA[orig_r]], row_nnz * sizeof(int));
                memcpy((*val_sub) + offset, &csrValA[csrRowPtrA[orig_r]], row_nnz * sizeof(MAT_VAL_TYPE));
            }
        }
    }
    else 
    {
        *idx_sub = NULL; 
        *val_sub = NULL;
    }
}

// 按列复用情况和 gamma、delta 阈值依次选择 64、16、8 行分组
void matrix_split(int rowA, int colA, MAT_PTR_TYPE nnzA,
                MAT_PTR_TYPE *csrRowPtrA, int *csrColIdxA, MAT_VAL_TYPE *csrValA,
                float gamma, float delta,
                int *rowA_64, MAT_PTR_TYPE *nnzA_64, MAT_PTR_TYPE **ptr_64, int **idx_64, MAT_VAL_TYPE **val_64, int **orig_row_64,
                int *rowA_16, MAT_PTR_TYPE *nnzA_16, MAT_PTR_TYPE **ptr_16, int **idx_16, MAT_VAL_TYPE **val_16, int **orig_row_16,
                int *rowA_8,  MAT_PTR_TYPE *nnzA_8,  MAT_PTR_TYPE **ptr_8,  int **idx_8,  MAT_VAL_TYPE **val_8,  int **orig_row_8) 
{
    // 阈值已经确定全部分组时，不再扫描列索引；仍由原函数生成子CSR
    int fixed_group = -1;
    if (gamma <= 0.0f) fixed_group = 0;
    else if (gamma >= 4.0f && delta <= 0.0f) fixed_group = 1;
    else if (gamma >= 4.0f && delta > 2.0f) fixed_group = 2;
    if (fixed_group >= 0)
    {
        int *group = (int *)malloc((size_t)rowA * sizeof(int));
        checkHostAlloc(group, (size_t)rowA * sizeof(int), "group");
        for (int r = 0; r < rowA; r++) group[r] = r;
        build_sub_csr(group, fixed_group == 0 ? rowA : 0, csrRowPtrA, csrColIdxA, csrValA,
                      rowA_64, nnzA_64, ptr_64, idx_64, val_64, orig_row_64);
        build_sub_csr(group, fixed_group == 1 ? rowA : 0, csrRowPtrA, csrColIdxA, csrValA,
                      rowA_16, nnzA_16, ptr_16, idx_16, val_16, orig_row_16);
        build_sub_csr(group, fixed_group == 2 ? rowA : 0, csrRowPtrA, csrColIdxA, csrValA,
                      rowA_8, nnzA_8, ptr_8, idx_8, val_8, orig_row_8);
        free(group);
        return;
    }

    int *group_64 = (int*)malloc(rowA * sizeof(int));
    checkHostAlloc(group_64, rowA * sizeof(int), "group_64");
    int *group_16 = (int*)malloc(rowA * sizeof(int));
    checkHostAlloc(group_16, rowA * sizeof(int), "group_16");
    int *group_8  = (int*)malloc(rowA * sizeof(int));
    checkHostAlloc(group_8, rowA * sizeof(int), "group_8");
    int count_64 = 0, count_16 = 0, count_8 = 0;

    char *flag = (char*)calloc(colA, sizeof(char));
    checkHostAlloc(flag, (colA) * (sizeof(char)), "flag");
    int *touched = (int*)malloc(colA * sizeof(int));
    checkHostAlloc(touched, colA * sizeof(int), "touched");

    int cache_size = rowA / 8 + (rowA % 8 != 0);
    int *count_cache = NULL;
    if (rowA >= 16 && gamma > 0.0f)
    {
        count_cache = (int *)malloc((size_t)cache_size * 2 * sizeof(int));
        checkHostAlloc(count_cache, (size_t)cache_size * 2 * sizeof(int), "count_cache");
        if (count_cache)
            for (size_t j = 0; j < (size_t)cache_size * 2; j++) count_cache[j] = -1;
    }

    int i = 0;
    
    while (i < rowA) 
    {
        int is_64 = 0;
        int cached_N16 = -1; 

        if (gamma <= 0.0) {
            is_64 = 1; 
        }
        else if (gamma < 4.0f && i + 64 <= rowA) 
        {
            int N64 = count_cols_cached(i, i + 64, csrRowPtrA, csrColIdxA, flag, touched, count_cache, cache_size);
            if (N64 > 0) 
            {
                float thresh_64 = gamma * N64;
                int S16 = 0;
                
                cached_N16 = count_cols_cached(i, i + 16, csrRowPtrA, csrColIdxA, flag, touched, count_cache, cache_size);
                S16 += cached_N16;
                if (S16 > thresh_64) { is_64 = 1; }
                
                if (!is_64) { 
                    S16 += count_cols_cached(i + 16, i + 32, csrRowPtrA, csrColIdxA, flag, touched, count_cache, cache_size); 
                    if (S16 > thresh_64) is_64 = 1; 
                }
                if (!is_64) { 
                    S16 += count_cols_cached(i + 32, i + 48, csrRowPtrA, csrColIdxA, flag, touched, count_cache, cache_size); 
                    if (S16 > thresh_64) is_64 = 1; 
                }
                if (!is_64) { 
                    S16 += count_cols_cached(i + 48, i + 64, csrRowPtrA, csrColIdxA, flag, touched, count_cache, cache_size); 
                    if (S16 > thresh_64) is_64 = 1; 
                }
            }
        }

        if (is_64) 
        {
            int end = MIN(i + 64, rowA); 
            for (int r = i; r < end; ++r) 
                group_64[count_64++] = r;
            i += 64; 
            continue; 
        }

        int is_16 = 0;
        
        if (delta <= 0.0) {
            is_16 = 1; 
        }
        else if (!(delta > 2.0f) && i + 16 <= rowA) 
        {
            int N16 = (cached_N16 != -1) ? cached_N16 : count_cols_cached(i, i + 16, csrRowPtrA, csrColIdxA, flag, touched, count_cache, cache_size);
            if (N16 > 0) 
            {
                float thresh_16 = delta * N16;
                int S8 = count_cols_cached(i, i + 8, csrRowPtrA, csrColIdxA, flag, touched, count_cache, cache_size);
                
                if (S8 >= thresh_16) { 
                    is_16 = 1; 
                } else {
                    S8 += count_cols_cached(i + 8, i + 16, csrRowPtrA, csrColIdxA, flag, touched, count_cache, cache_size);
                    if (S8 >= thresh_16) is_16 = 1;
                }
            }
        }

        if (is_16) 
        {
            int end = MIN(i + 16, rowA);
            for (int r = i; r < end; ++r) 
                group_16[count_16++] = r;
            
            i += 16; 
            continue;
        }

        int end8 = MIN(i + 8, rowA);
        for (int r = i; r < end8; ++r) 
            group_8[count_8++] = r;
        
        i += 8; 
    }

    free(count_cache);


    build_sub_csr(group_64, count_64, csrRowPtrA, csrColIdxA, csrValA, rowA_64, nnzA_64, ptr_64, idx_64, val_64, orig_row_64);
    build_sub_csr(group_16, count_16, csrRowPtrA, csrColIdxA, csrValA, rowA_16, nnzA_16, ptr_16, idx_16, val_16, orig_row_16);
    build_sub_csr(group_8,  count_8,  csrRowPtrA, csrColIdxA, csrValA, rowA_8,  nnzA_8,  ptr_8,  idx_8,  val_8,  orig_row_8);

    free(flag);
    free(touched);
    free(group_64);
    free(group_16);
    free(group_8);
}