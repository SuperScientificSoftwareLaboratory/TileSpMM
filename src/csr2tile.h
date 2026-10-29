#include "common.h"
#include "utils.h"
#include "encode.h"
#include "format.h"
#include "tilespmm_cuda.h"
#ifdef TFLOAT
#include "../lib/prefetch_tf32.h"
#else
#include "../lib/prefetch_fp16.h"
#endif

// =========================================================
// Step 1: 统计每个行块的 Tile 数量及转换所需的临时空间
// =========================================================
void convert_step1(Tile_matrix *matrix, int rowA, int colA, MAT_PTR_TYPE nnzA,
                   MAT_PTR_TYPE *csrRowPtrA, int *csrColIdxA, MAT_VAL_TYPE *csrValA, 
                   int BM, int BN, char *flag_g, int *valid_cols_g,
                   int *nnz_temp_max, int *tile_count_max)
{
    int tilem = matrix->tilem;
    int *tile_ptr = matrix->tile_ptr;
    int max_nnz = 0, max_tiles = 0;
    
#pragma omp parallel reduction(max:max_nnz, max_tiles)
    {
        int tid = omp_get_thread_num();
        char *flag = flag_g + tid * colA;
        int *valid_cols = valid_cols_g + tid * colA;

#pragma omp for schedule(dynamic, 32)
        for (int blki = 0; blki < tilem; blki++)
        {
            int valid_count = 0;
            int start = blki * BM;
            int end = (blki == tilem - 1) ? rowA : (blki + 1) * BM;
            
            for (MAT_PTR_TYPE j = csrRowPtrA[start]; j < csrRowPtrA[end]; j++) {
                int col = csrColIdxA[j];
                if (flag[col] == 0) {
                    flag[col] = 1;
                    valid_cols[valid_count++] = col;
                }
            }

            if (valid_count == 0) tile_ptr[blki] = 0;
            else tile_ptr[blki] = (valid_count + BN - 1) / BN;
            
            int row_nnz = csrRowPtrA[end] - csrRowPtrA[start];
            if (row_nnz > max_nnz) max_nnz = row_nnz;
            if (tile_ptr[blki] > max_tiles) max_tiles = tile_ptr[blki];
            for(int k = 0; k < valid_count; k++) flag[valid_cols[k]] = 0;
        }
    }
    *nnz_temp_max = max_nnz;
    *tile_count_max = max_tiles;
}

// =========================================================
// Step 2: 映射列索引、统计每个 Tile 的非零元数量
// =========================================================
void convert_step2(Tile_matrix *matrix, unsigned char *tile_csr_ptr,
                   int rowA, int colA, MAT_PTR_TYPE nnzA,
                   MAT_PTR_TYPE *csrRowPtrA, int *csrColIdxA, MAT_VAL_TYPE *csrValA, 
                   int BM, int BN, int tile_count_temp,
                   char *flag_g, int *valid_cols_g, int *col_to_rank_g, int *tile_count_g)
{
    int tilem = matrix->tilem;
    int *tile_ptr = matrix->tile_ptr;
    int *tile_nnz = matrix->tile_nnz;
    int *tile_colidx = matrix->tile_colidx;
#pragma omp parallel
    {

#pragma omp for schedule(dynamic, 32)
        for (int blki = 0; blki < tilem; blki++)
        {
            int thread_id = omp_get_thread_num();
            char *flag = flag_g + (size_t)thread_id * colA;
            int *valid_cols = valid_cols_g + (size_t)thread_id * colA;
            int *col_to_rank = col_to_rank_g + (size_t)thread_id * colA;
            int *nnz_temp = tile_count_g + (size_t)thread_id * tile_count_temp;

            int pre_tile = tile_ptr[blki]; 
            int tiles_this_row = tile_ptr[blki+1] - tile_ptr[blki]; 
            if (tiles_this_row == 0) continue;
            unsigned char *ptr_per_tile = tile_csr_ptr + (size_t)pre_tile * BM;
            
            memset(nnz_temp, 0, tiles_this_row * sizeof(int));
            memset(ptr_per_tile, 0, tiles_this_row * BM * sizeof(unsigned char));
            int rowlen = blki == tilem - 1 ? rowA - (tilem - 1) * BM : BM;
            int start = blki * BM;
            int end = blki == tilem - 1 ? rowA : (blki + 1) * BM;

            int valid_count = 0;
            for (int j = csrRowPtrA[start]; j < csrRowPtrA[end]; j++) {
                int col = csrColIdxA[j];
                if (flag[col] == 0) {
                    flag[col] = 1;
                    valid_cols[valid_count++] = col;
                }
            }
            
            if (valid_count < 128) {
                for (int idx = 1; idx < valid_count; idx++) {
                    int key = valid_cols[idx];
                    int j = idx - 1;
                    while (j >= 0 && valid_cols[j] > key) {
                        valid_cols[j + 1] = valid_cols[j];
                        j = j - 1;
                    }
                    valid_cols[j + 1] = key;
                }
            } else {
                quick_sort_key(valid_cols, valid_count);
            }

            int last_valid_col = (valid_count > 0) ? valid_cols[valid_count - 1] : 0;
            
            for(int k = 0; k < valid_count; k++) {
                int original_col = valid_cols[k];
                col_to_rank[original_col] = k; 
                int t_idx = k / BN; 
                int t_col = k % BN;
                if (t_idx < tiles_this_row) tile_colidx[(pre_tile + t_idx) * BN + t_col] = original_col;
                flag[original_col] = 0; 
            }
            
            for(int k = valid_count; k < tiles_this_row * BN; k++) {
                 int t_idx = k / BN;
                 int t_col = k % BN;
                 tile_colidx[(pre_tile + t_idx) * BN + t_col] = last_valid_col; 
            }

            for (int ri = 0; ri < rowlen; ri++) {
                for (int j = csrRowPtrA[start + ri]; j < csrRowPtrA[start + ri + 1]; j++) {
                    int original_col = csrColIdxA[j];
                    int rank = col_to_rank[original_col]; 
                    int tile_local_idx = rank / BN;
                    nnz_temp[tile_local_idx]++;
                    ptr_per_tile[tile_local_idx * BM + ri]++;
                }
            }

            for (int blkj = 0; blkj < tiles_this_row; blkj++) {
                tile_nnz[pre_tile + blkj] = nnz_temp[blkj];
            }
        }
    }

}

// =========================================================
// Step 3: 根据 BM 执行对应块大小的数据排布
// =========================================================
void convert_step3(Tile_matrix *matrix, unsigned char *tile_csr_ptr,
                   int nnz_temp_max, int tile_count_temp,
                   int rowA, int colA, MAT_PTR_TYPE nnzA,
                   MAT_PTR_TYPE *csrRowPtrA, int *csrColIdxA, MAT_VAL_TYPE *csrValA, 
                   int BM, int BN, char *flag_g, int *valid_cols_g,
                   int *col_to_rank_g, int *tile_count_g)
{
    int tilem = matrix->tilem;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_nnz = matrix->tile_nnz;
    MAT_VAL_TYPE *Blockdense_Val = matrix->Blockdense_Val;
    
    unsigned thread = omp_get_max_threads();
    unsigned char *csr_colidx_temp_g = (unsigned char *)malloc((size_t)thread * nnz_temp_max * sizeof(unsigned char));
    checkHostAlloc(csr_colidx_temp_g, (size_t)thread * nnz_temp_max * sizeof(unsigned char), "csr_colidx_temp_g");
    MAT_VAL_TYPE *csr_val_temp_g = (MAT_VAL_TYPE *)malloc((size_t)thread * nnz_temp_max * sizeof(MAT_VAL_TYPE));
    checkHostAlloc(csr_val_temp_g, (size_t)thread * nnz_temp_max * sizeof(MAT_VAL_TYPE), "csr_val_temp_g");
    

#pragma omp parallel
    {

#pragma omp for schedule(dynamic, 32)
        for (int blki = 0; blki < tilem; blki++)
        {
            int thread_id = omp_get_thread_num();
            unsigned char *csr_colidx_temp = csr_colidx_temp_g + (size_t)thread_id * nnz_temp_max;
            MAT_VAL_TYPE *csr_val_temp = csr_val_temp_g + (size_t)thread_id * nnz_temp_max;
            int *tile_count = tile_count_g + (size_t)thread_id * tile_count_temp;
            
            char *flag = flag_g + (size_t)thread_id * colA;
            int *valid_cols = valid_cols_g + (size_t)thread_id * colA;
            int *col_to_rank = col_to_rank_g + (size_t)thread_id * colA;
            
            int tilenum_per_row = tile_ptr[blki + 1] - tile_ptr[blki];
            if (tilenum_per_row == 0) continue;
            memset(tile_count, 0, tilenum_per_row * sizeof(int));
            int rowlen = blki == tilem - 1 ? rowA - (tilem - 1) * BM : BM;
            int start = blki * BM;
            int end = blki == tilem - 1 ? rowA : (blki + 1) * BM;

            int valid_count = 0;
            int pre_tile = tile_ptr[blki];
            // BM=8的每个tile行独占输出区间，先清零再按原顺序累加
            if (BM == 8)
                memset(Blockdense_Val + (size_t)pre_tile * BM * BN, 0, (size_t)tilenum_per_row * BM * BN * sizeof(MAT_VAL_TYPE));
            
            for (int t = 0; t < tilenum_per_row; t++) {
                for (int c = 0; c < BN; c++) {
                    int col = matrix->tile_colidx[(pre_tile + t) * BN + c];
                    if (flag[col] == 0) {
                        col_to_rank[col] = t * BN + c;
                        flag[col] = 1;
                        valid_cols[valid_count++] = col;
                    }
                }
            }
            // 清理本次访问过的列标记
            for (int k = 0; k < valid_count; k++) {
                flag[valid_cols[k]] = 0; 
            }

            for (int blkj = csrRowPtrA[start]; blkj < csrRowPtrA[end]; blkj++) {
                int original_col = csrColIdxA[blkj];
                int rank = col_to_rank[original_col];
                int tile_local_idx = rank / BN; 
                int inner_col = rank % BN; 

                int tile_id = tile_ptr[blki] + tile_local_idx;
                int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr[blki]];
                
                csr_val_temp[pre_nnz + tile_count[tile_local_idx]] = csrValA[blkj];
                csr_colidx_temp[pre_nnz + tile_count[tile_local_idx]] = inner_col; 
                tile_count[tile_local_idx]++;
            }

            for (int bi = 0; bi < tilenum_per_row; bi++)
            {
                int tile_id = tile_ptr[blki] + bi;
                int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr[blki]];
                int nnztmp = tile_nnz[tile_id + 1] - tile_nnz[tile_id];
                
                size_t offset = (size_t)tile_id * BM * BN;
                unsigned char *ptr_temp = tile_csr_ptr + (size_t)tile_id * BM;
                
                int row_offsets[65]; // 最大 64 偏移支持
                row_offsets[0] = 0;
                if(BM == 64) 
                {
                    for(int r = 0; r < BM; r++)
                     {
                        int count = (r < rowlen) ? ptr_temp[r] : 0;
                        row_offsets[r+1] = row_offsets[r] + count;
                    }
                } 
                else 
                {
                    exclusive_scan_char(ptr_temp, BM); 
                }

                if (BM == 8) 
                {
                    for (int ri = 0; ri < rowlen; ri++) {
                        int nnz_end = ri == rowlen - 1 ? nnztmp : ptr_temp[ri + 1];
                        for (int j = ptr_temp[ri]; j < nnz_end; j++) {
                            int col = csr_colidx_temp[pre_nnz + j]; 
                            float val = csr_val_temp[pre_nnz + j];
#ifdef TFLOAT
                            int lane_id = (ri << 2) + (col & 3);
                            int slot_id = col >> 2;
                            int swizzled_idx = (lane_id << 1) + slot_id;
#else
                            int lane_id = (ri << 2) + ((col >> 1) & 3);
                            int slot_id = ((col >> 3) << 1) + (col & 1);
                            int swizzled_idx = (lane_id << 2) + slot_id;
#endif
                            Blockdense_Val[offset + swizzled_idx] += val;
                        }
                    }
                } 
                else if (BM == 16) 
                {
                    MAT_VAL_TYPE temp_dense_16xK[16 * 16]; // 最大 16x16
                    memset(temp_dense_16xK, 0, 16 * BN * sizeof(MAT_VAL_TYPE));

                    for (int ri = 0; ri < rowlen; ri++) {
                        int nnz_end = ri == rowlen - 1 ? nnztmp : ptr_temp[ri + 1];
                        for (int j = ptr_temp[ri]; j < nnz_end; j++) {
                            temp_dense_16xK[ri * BN + csr_colidx_temp[pre_nnz + j]] += csr_val_temp[pre_nnz + j];
                        }
                    }

                    for (int lane = 0; lane < 32; lane++) {
#ifdef TFLOAT
                        int r_idx = lane >> 2;   
                        int c_idx = lane & 3;     
                        size_t dst_idx = offset + lane * 4; 
                        Blockdense_Val[dst_idx + 0] = temp_dense_16xK[r_idx * BN + c_idx];
                        Blockdense_Val[dst_idx + 1] = temp_dense_16xK[(r_idx + 8) * BN + c_idx];
                        Blockdense_Val[dst_idx + 2] = temp_dense_16xK[r_idx * BN + c_idx + 4];
                        Blockdense_Val[dst_idx + 3] = temp_dense_16xK[(r_idx + 8) * BN + c_idx + 4];
#else
                        int row0 = lane >> 2;    
                        int col_base = (lane & 3) << 1; 
                        size_t dst_idx = offset + lane * 8; 
                        Blockdense_Val[dst_idx + 0] = temp_dense_16xK[(row0) * BN + (col_base)];
                        Blockdense_Val[dst_idx + 1] = temp_dense_16xK[(row0) * BN + (col_base + 1)];
                        Blockdense_Val[dst_idx + 2] = temp_dense_16xK[(row0 + 8) * BN + (col_base)];
                        Blockdense_Val[dst_idx + 3] = temp_dense_16xK[(row0 + 8) * BN + (col_base + 1)];
                        Blockdense_Val[dst_idx + 4] = temp_dense_16xK[(row0) * BN + (col_base + 8)];
                        Blockdense_Val[dst_idx + 5] = temp_dense_16xK[(row0) * BN + (col_base + 9)];
                        Blockdense_Val[dst_idx + 6] = temp_dense_16xK[(row0 + 8) * BN + (col_base + 8)];
                        Blockdense_Val[dst_idx + 7] = temp_dense_16xK[(row0 + 8) * BN + (col_base + 9)];
#endif
                    }
                }
                else if (BM == 64) 
                {
                    // 一次按原坐标顺序累加64行稠密块，再按原布局写出各个MMA子块
                    MAT_VAL_TYPE temp_dense_64xK[64 * 32];
                    memset(temp_dense_64xK, 0, (size_t)BM * BN * sizeof(MAT_VAL_TYPE));
                    for (int ri = 0; ri < rowlen; ri++)
                        for (int j = row_offsets[ri]; j < row_offsets[ri + 1]; j++)
                            temp_dense_64xK[ri * BN + csr_colidx_temp[pre_nnz + j]] += csr_val_temp[pre_nnz + j];
                    size_t global_sub_idx = 0;
#ifdef TFLOAT
                    int num_subtiles_m = (BM + 16 - 1) / 16; 
                    int num_subtiles_k = (BN + 8 - 1) / 8;
                    for (int sub_m = 0; sub_m < num_subtiles_m; sub_m++) {
                        int sub_row_start = sub_m * 16;
                        int sub_row_end = (sub_m + 1) * 16;
                        if (sub_row_start >= rowlen) sub_row_end = sub_row_start; 
                        else if (sub_row_end > rowlen) sub_row_end = rowlen;

                        for (int sub_k = 0; sub_k < num_subtiles_k; sub_k++) {
                            int sub_col_start = sub_k * 8;
                            int sub_col_end = (sub_k + 1) * 8;
                            size_t sub_global_offset = offset + global_sub_idx * (size_t)(16 * 8);
                            global_sub_idx++; 
                            for (int lane = 0; lane < 32; lane++) {
                                int row0 = lane >> 2; int col0 = lane & 3;
                                size_t dst_idx = sub_global_offset + (size_t)(lane * 4); 
                                Blockdense_Val[dst_idx + 0] = temp_dense_64xK[(sub_row_start + (row0)) * BN + sub_col_start + (col0)];
                                Blockdense_Val[dst_idx + 1] = temp_dense_64xK[(sub_row_start + (row0 + 8)) * BN + sub_col_start + (col0)];
                                Blockdense_Val[dst_idx + 2] = temp_dense_64xK[(sub_row_start + (row0)) * BN + sub_col_start + (col0 + 4)];
                                Blockdense_Val[dst_idx + 3] = temp_dense_64xK[(sub_row_start + (row0 + 8)) * BN + sub_col_start + (col0 + 4)];
                            }
                        }
                    }
#else
                    int num_subtiles_m = (BM + 16 - 1) / 16; 
                    int num_subtiles_k = (BN + 16 - 1) / 16;
                    for (int sub_m = 0; sub_m < num_subtiles_m; sub_m++) {
                        int sub_row_start = sub_m * 16;
                        int sub_row_end = (sub_m + 1) * 16;
                        if (sub_row_start >= rowlen) sub_row_end = sub_row_start; 
                        else if (sub_row_end > rowlen) sub_row_end = rowlen;

                        for (int sub_k = 0; sub_k < num_subtiles_k; sub_k++) {
                            int sub_col_start = sub_k * 16;
                            int sub_col_end = (sub_k + 1) * 16;
                            size_t sub_global_offset = offset + global_sub_idx * (size_t)(16 * 16);
                            global_sub_idx++; 
                            for (int lane = 0; lane < 32; lane++) {
                                int row0 = lane >> 2; int col_base = (lane & 3) << 1; 
                                size_t dst_idx = sub_global_offset + (size_t)(lane * 8); 
                                Blockdense_Val[dst_idx + 0] = temp_dense_64xK[(sub_row_start + (row0)) * BN + sub_col_start + (col_base)];
                                Blockdense_Val[dst_idx + 1] = temp_dense_64xK[(sub_row_start + (row0)) * BN + sub_col_start + (col_base + 1)];
                                Blockdense_Val[dst_idx + 2] = temp_dense_64xK[(sub_row_start + (row0 + 8)) * BN + sub_col_start + (col_base)];
                                Blockdense_Val[dst_idx + 3] = temp_dense_64xK[(sub_row_start + (row0 + 8)) * BN + sub_col_start + (col_base + 1)];
                                Blockdense_Val[dst_idx + 4] = temp_dense_64xK[(sub_row_start + (row0)) * BN + sub_col_start + (col_base + 8)];
                                Blockdense_Val[dst_idx + 5] = temp_dense_64xK[(sub_row_start + (row0)) * BN + sub_col_start + (col_base + 9)];
                                Blockdense_Val[dst_idx + 6] = temp_dense_64xK[(sub_row_start + (row0 + 8)) * BN + sub_col_start + (col_base + 8)];
                                Blockdense_Val[dst_idx + 7] = temp_dense_64xK[(sub_row_start + (row0 + 8)) * BN + sub_col_start + (col_base + 9)];
                            }
                        }
                    }
#endif
                }
            }
        } 
    }
    free(csr_colidx_temp_g); 
    free(csr_val_temp_g);
}


// 按切分阈值生成任务行号和片段起点，长行块标记为原子累加任务
void tile_pretreatment(Tile_matrix *matrix, int *rowblkblock,
                       unsigned int **blkcoostylerowidx, int **blkcoostylerowidx_colstart, 
                       int prefetch_smem_th)
{
    int tilem = matrix->tilem;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int rowblkblock_tmp = 0;
    
#pragma omp parallel for reduction(+:rowblkblock_tmp)
    for (int blki = 0; blki < tilem; blki++) 
    {
        int balancenumblk = tile_ptr[blki + 1] - tile_ptr[blki];
        if (balancenumblk <= prefetch_smem_th) rowblkblock_tmp++;
        else rowblkblock_tmp += (int)ceil((double)balancenumblk / (double)prefetch_smem_th);
    }
    *rowblkblock = rowblkblock_tmp;

    *blkcoostylerowidx = (unsigned int *)malloc(sizeof(unsigned int) * *rowblkblock);
    checkHostAlloc(*blkcoostylerowidx, sizeof(unsigned int) * *rowblkblock, "blkcoostylerowidx");

    *blkcoostylerowidx_colstart = (int *)malloc(sizeof(int) * (*rowblkblock + 1));
    checkHostAlloc(*blkcoostylerowidx_colstart, sizeof(int) * (*rowblkblock + 1), "blkcoostylerowidx_colstart");
    
    (*blkcoostylerowidx_colstart)[0] = tile_ptr[0];
    
    int rowblkblockcnt = 0;
    for (int blki = 0; blki < tilem; blki++) 
    {
        int balancenumblk = tile_ptr[blki + 1] - tile_ptr[blki];
        int real_start_row = matrix->orig_tile_row_idx[blki];
        if (balancenumblk <= prefetch_smem_th) 
        {
            (*blkcoostylerowidx)[rowblkblockcnt] = real_start_row;
            (*blkcoostylerowidx_colstart)[rowblkblockcnt + 1] = tile_ptr[blki + 1];
            rowblkblockcnt++;
        } 
        else 
        {
            int numblklocal = (int)ceil((double)balancenumblk / (double)prefetch_smem_th);
            int lenblklocal = (int)ceil((double)balancenumblk / (double)numblklocal);
            for (int iii = 0; iii < numblklocal; iii++) 
            {
                (*blkcoostylerowidx)[rowblkblockcnt] = real_start_row | 0x80000000;
                if (iii == numblklocal - 1)
                    (*blkcoostylerowidx_colstart)[rowblkblockcnt + 1] = tile_ptr[blki] + balancenumblk;
                else
                    (*blkcoostylerowidx_colstart)[rowblkblockcnt + 1] = tile_ptr[blki] + (iii + 1) * lenblklocal;
                rowblkblockcnt++;
            }
        }
    }
}

// 根据行块长度分布和资源代价，筛选四个待实测的切分阈值
void get_optimal_prefetch_th(int BM, int colB, int is_tf32, int tilem, int* tile_ptr, int* top4_T, const cudaDeviceProp& deviceProp)
{
    // 根据设备上限和预编译核函数资源估计并发能力
    const int SM = prefetch_device_info.sm_count;
    const int TH_SM = prefetch_device_info.threads_per_sm;
    const size_t SMEM_SM = prefetch_device_info.shared_bytes_per_sm;
    const int BLK_SM_HW = prefetch_device_info.blocks_per_sm;
    const int T_MAX = 128;
    const int T_MIN = 8;
    
    int group = BM == 8 ? 0 : (BM == 16 ? 1 : (BM == 64 ? 2 : -1));
#ifdef TFLOAT
    const int compiled_tf32 = 1;
#else
    const int compiled_tf32 = 0;
#endif
    if (group < 0 || is_tf32 != compiled_tf32)
    {
        fprintf(stderr, "Invalid kernel selection for prefetch resources.\n");
        exit(EXIT_FAILURE);
    }
    const PrefetchKernelInfo &kernel = prefetch_kernel_info[group * 2 + (colB != 32)];
    int th_blk = kernel.threads;
    int reg_th = kernel.registers_per_thread;
    size_t smem_blk = kernel.shared_bytes;

    int blk_reg = prefetch_device_info.registers_per_sm / (reg_th * th_blk);
    int blk_smem = smem_blk > 0 ? (SMEM_SM / smem_blk) : BLK_SM_HW;

    int blk_sm = TH_SM / th_blk;
    if (blk_reg < blk_sm) blk_sm = blk_reg;
    if (blk_smem < blk_sm) blk_sm = blk_smem;
    if (blk_sm > BLK_SM_HW) blk_sm = BLK_SM_HW;
    if (blk_sm <= 0)
    {
        fprintf(stderr, "The prepared kernel resources do not fit the selected GPU.\n");
        exit(EXIT_FAILURE);
    }

    // 并发容量：每个 SM 可驻留的线程块数乘以 SM 数量
    float cap = (float)(blk_sm * SM);

    // L2 拥堵权重：每次迭代访存量乘以并发容量，再除以 L2 容量
    int S_iter = kernel.iteration_bytes;
    float lambda_L2 = (float)(S_iter * cap) / (float)deviceProp.l2CacheSize;

    // 原子操作权重由已有平台数据校准，不由准备脚本测量
    float lambda_atom = 0.01f;
    if (colB == 32)
    {
        if (BM == 64) lambda_atom = is_tf32 ? 0.020f : 0.019f;
        else if (BM == 16) lambda_atom = is_tf32 ? 0.00830f : 0.00844f;
        else lambda_atom = is_tf32 ? 0.00674f : 0.00360f;
    }
    else
    {
        if (BM == 16) lambda_atom = is_tf32 ? 0.01048f : 0.01163f;
        else if (BM == 8) lambda_atom = is_tf32 ? 0.00960f : 0.00826f;
    }
    // 按 32 列归一化原子权重，保持原来的运算顺序
    float atom_per_col = lambda_atom * (1.0f / 32.0f);

    int max_len = 0;
    long long total_len = 0;

    for (int i = 0; i < tilem; i++)
    {
        int len = tile_ptr[i + 1] - tile_ptr[i];
        if (len == 0) continue;

        if (len > max_len) max_len = len;
        total_len += len;
    }

    if (max_len == 0 || tilem == 0) {
        for(int i = 0; i < 4; i++) top4_T[i] = 32;
        return;
    }

    int *hist = (int *)calloc(max_len + 1, sizeof(int));
    checkHostAlloc(hist, (max_len + 1) * (sizeof(int)), "hist");

    // 按行块的片段数量建立直方图
    int distinct_len = 0;
    for (int i = 0; i < tilem; i++)
    {
        int len = tile_ptr[i + 1] - tile_ptr[i];
        if (len > 0)
        {
            if (hist[len] == 0) distinct_len++;
            hist[len]++;
        }
    }

    int *active_len = NULL;
    int active_count = max_len;
    if (max_len > 256 && distinct_len < max_len / 2)
    {
        active_len = (int *)malloc((size_t)distinct_len * sizeof(int));
        checkHostAlloc(active_len, (size_t)distinct_len * sizeof(int), "active_len");
        if (active_len)
        {
            active_count = 0;
            for (int len = 1; len <= max_len; len++)
                if (hist[len]) active_len[active_count++] = len;
        }
    }

    float avg_len = (float)total_len / tilem; 
    float atom_pen = atom_per_col * (float)colB; 
    
    // 保留长尾保护：极长行块出现时降低原子项权重
    if (max_len > avg_len * 5.0f) atom_pen *= 0.04f; 

    int Tmax = (max_len + T_MIN - 1) / T_MIN * T_MIN;
    Tmax = (Tmax < T_MAX) ? Tmax : T_MAX;

    CostCand cands[128];
    int cand_cnt = 0;
    long long previous_total_chunk = -1;

    for (int TH = 8; TH <= Tmax; TH += 8)
    {
        long long atom_ops = 0;
        int max_chunk = 0;
        long long total_chunk = 0;
        float l2_penalty = 0.0f;

        for (int index = 0; index < active_count; index++)
        {
            int len = active_len ? active_len[index] : index + 1;
            int cnt = hist[len];
            if (cnt == 0) continue;

            // 先确定任务数，再计算均匀切分后单个任务的最大片段数
            int blk = (len + TH - 1) / TH; 
            int blk_task = (len + blk - 1) / blk;
            atom_ops += (long long)(blk - 1) * cnt;
            total_chunk += (long long)blk * cnt;

            if (blk_task > max_chunk) max_chunk = blk_task;

            if (blk > 1) 
            {
                float l2_miss = 1.0f - (1.0f / (float)blk);
                l2_penalty += l2_miss * cnt;
            }
        }

        // 任务总数相同的后续阈值不再重复评估
        if (total_chunk == previous_total_chunk) continue;
        previous_total_chunk = total_chunk;

        // 32 列的 8/16 行核函数每块打包四个任务，其余按输出列网格展开
        long long total_block = 0;
        if (colB == 32) 
        {
            if (BM == 8 || BM == 16) total_block = (total_chunk + 3) / 4;
            else total_block = total_chunk;
        } 
        else 
        {
            int num_blk_x = (colB + 63) / 64; 
            total_block = total_chunk * num_blk_x;
        }

        float cost_base = (float)total_block / cap * TH;    // 吞吐代价
        float cost_tail = (float)max_chunk;                // 尾部代价
        
        float cost_l2 = l2_penalty * lambda_L2;            // L2 访存代价

        // 合并吞吐、尾部、L2 和原子代价，保持原来的浮点加法顺序
        float cost_comp = cost_base + cost_tail + cost_l2;
        float cost_atom = atom_pen * atom_ops;            // 原子累加代价
        
        float cost = cost_comp + cost_atom;
        
        cands[cand_cnt].TH = TH;
        cands[cand_cnt].cost = cost;
        cand_cnt++;
    }
    
    // 返回代价最小的四个候选；代价相同时保持原先顺序
    for (int i = 0; i < cand_cnt - 1; i++) 
    {
        for (int j = 0; j < cand_cnt - i - 1; j++) 
        {
            if (cands[j].cost > cands[j+1].cost) 
            {
                CostCand temp = cands[j];
                cands[j] = cands[j+1];
                cands[j+1] = temp;
            }
        }
    }

    for (int i = 0; i < 4; i++) 
    {
        if (i < cand_cnt) top4_T[i] = cands[i].TH;
        else top4_T[i] = cands[0].TH;
    }


    free(active_len);
    free(hist);
}
// 释放缓存中未被当前矩阵持有的调度，避免重复释放
void free_prefetch_cache(Tile_matrix *mat, Prefetch_cache *cache)
{
    for (int i = 0; i < 4; i++)
    {
        if (cache->entries[i].TH && cache->entries[i].rows != mat->blkcoostylerowidx)
        {
            free(cache->entries[i].rows);
            free(cache->entries[i].colstart);
        }
    }
}

// 复用已缓存的调度；没有命中时构建，再上传到对应工作流
void update_metadata(Tile_matrix *mat, int prefetch_smem_th,
                                    unsigned int *d_blkcoostylerowidx, 
                                    int *d_blkcoostylerowidx_colstart,
                                    cudaStream_t stream, Prefetch_cache *cache = NULL)
{
    if (mat->tilenum == 0) return;

    bool current_cached = false;
    int found = -1, empty = -1;
    if (cache)
    {
        for (int i = 0; i < 4; i++)
        {
            if (cache->entries[i].TH && cache->entries[i].rows == mat->blkcoostylerowidx)
                current_cached = true;
            if (cache->entries[i].TH == prefetch_smem_th) found = i;
            if (!cache->entries[i].TH && empty < 0) empty = i;
        }
    }
    if (!current_cached)
    {
        if (mat->blkcoostylerowidx) free(mat->blkcoostylerowidx);
        if (mat->blkcoostylerowidx_colstart) free(mat->blkcoostylerowidx_colstart);
    }

    int rowblkblock = 0;
    unsigned int *blkcoostylerowidx = NULL;
    int *blkcoostylerowidx_colstart = NULL;
    if (found >= 0)
    {
        rowblkblock = cache->entries[found].rowblkblock;
        blkcoostylerowidx = cache->entries[found].rows;
        blkcoostylerowidx_colstart = cache->entries[found].colstart;
    }
    else
    {
        tile_pretreatment(mat, &rowblkblock, &blkcoostylerowidx, &blkcoostylerowidx_colstart, prefetch_smem_th);
        size_t bytes = (size_t)rowblkblock * (sizeof(unsigned int) + sizeof(int)) + sizeof(int);
        const size_t cache_limit = (size_t)16 * 1024 * 1024;
        if (cache && empty >= 0 && bytes <= cache_limit - cache->bytes)
        {
            cache->entries[empty].TH = prefetch_smem_th;
            cache->entries[empty].rowblkblock = rowblkblock;
            cache->entries[empty].rows = blkcoostylerowidx;
            cache->entries[empty].colstart = blkcoostylerowidx_colstart;
            cache->bytes += bytes;
        }
    }
    mat->rowblkblock = rowblkblock;
    mat->blkcoostylerowidx = blkcoostylerowidx;
    mat->blkcoostylerowidx_colstart = blkcoostylerowidx_colstart;

    checkCuda(cudaMemcpyAsync(d_blkcoostylerowidx, blkcoostylerowidx, rowblkblock * sizeof(unsigned int), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync d_blkcoostylerowidx");
    checkCuda(cudaMemcpyAsync(d_blkcoostylerowidx_colstart, blkcoostylerowidx_colstart, (rowblkblock + 1) * sizeof(int), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync d_blkcoostylerowidx_colstart");
}

// 释放三种行块转换共用的临时数组并清空工作区
void free_tile_workspace(Tile_workspace *workspace)
{
    free(workspace->flag);
    free(workspace->valid_cols);
    free(workspace->col_to_rank);
    free(workspace->tile_count);
    memset(workspace, 0, sizeof(Tile_workspace));
}

// 依次统计、映射和排布，将一组子 CSR 转换为对应的 Tile 格式
void Tile_create(Tile_matrix *matrix, 
                     int rowA, int colA, MAT_PTR_TYPE nnzA, int colB, 
                     MAT_PTR_TYPE *csrRowPtrA, int *csrColIdxA, MAT_VAL_TYPE *csrValA,
                     int *orig_row_sub,
                     int BM, int BN, Tile_workspace *workspace) 
{
    matrix->tilem = (rowA + BM - 1) / BM;
    matrix->tilen = (colA + BN - 1) / BN;
    matrix->tile_ptr = (int *)malloc((size_t)(matrix->tilem + 1) * sizeof(int));
    checkHostAlloc(matrix->tile_ptr, (size_t)(matrix->tilem + 1) * sizeof(int), "matrix->tile_ptr");
    matrix->tile_ptr[matrix->tilem] = 0;
    matrix->orig_tile_row_idx = (int *)malloc((size_t)matrix->tilem * sizeof(int));
    checkHostAlloc(matrix->orig_tile_row_idx, (size_t)matrix->tilem * sizeof(int), "matrix->orig_tile_row_idx");
    for (int blki = 0; blki < matrix->tilem; blki++) {
        matrix->orig_tile_row_idx[blki] = orig_row_sub[blki * BM];
    }
    unsigned thread = omp_get_max_threads();
    if (!workspace->flag || workspace->thread != thread || workspace->colA != colA)
    {
        free_tile_workspace(workspace);
        workspace->thread = thread;
        workspace->colA = colA;
        workspace->flag = (char *)calloc((size_t)thread * colA, sizeof(char));
        checkHostAlloc(workspace->flag, ((size_t)thread * colA) * (sizeof(char)), "workspace->flag");
        workspace->valid_cols = (int *)malloc((size_t)thread * colA * sizeof(int));
        checkHostAlloc(workspace->valid_cols, (size_t)thread * colA * sizeof(int), "workspace->valid_cols");
        workspace->col_to_rank = (int *)malloc((size_t)thread * colA * sizeof(int));
        checkHostAlloc(workspace->col_to_rank, (size_t)thread * colA * sizeof(int), "workspace->col_to_rank");
    }
    char *flag_g = workspace->flag;
    int *valid_cols_g = workspace->valid_cols;
    int *col_to_rank_g = workspace->col_to_rank;
    int nnz_temp = 0, tile_count_temp = 0;
    convert_step1(matrix, rowA, colA, nnzA, csrRowPtrA, csrColIdxA, csrValA, BM, BN,
                  flag_g, valid_cols_g, &nnz_temp, &tile_count_temp);
    exclusive_scan(matrix->tile_ptr, matrix->tilem + 1);
    matrix->tilenum = matrix->tile_ptr[matrix->tilem];
    int tilenum = matrix->tilenum;

    matrix->tile_nnz = (int *)malloc((size_t)(tilenum + 1) * sizeof(int));
    checkHostAlloc(matrix->tile_nnz, (size_t)(tilenum + 1) * sizeof(int), "matrix->tile_nnz");
    matrix->tile_nnz[tilenum] = 0;
    
    matrix->tile_colidx = (int *)malloc((size_t)tilenum * BN * sizeof(int));
    checkHostAlloc(matrix->tile_colidx, (size_t)tilenum * BN * sizeof(int), "matrix->tile_colidx");
    
    unsigned char *tile_csr_ptr = (unsigned char *)malloc((size_t)tilenum * BM * sizeof(unsigned char));
    checkHostAlloc(tile_csr_ptr, (size_t)tilenum * BM * sizeof(unsigned char), "tile_csr_ptr");

    if (tile_count_temp == 0) tile_count_temp = 1;
    if (workspace->tile_count_capacity < tile_count_temp)
    {
        free(workspace->tile_count);
        workspace->tile_count = (int *)malloc((size_t)thread * tile_count_temp * sizeof(int));
        checkHostAlloc(workspace->tile_count, (size_t)thread * tile_count_temp * sizeof(int), "workspace->tile_count");
        workspace->tile_count_capacity = tile_count_temp;
    }
    int *tile_count_g = workspace->tile_count;
    tile_count_temp = workspace->tile_count_capacity;

    convert_step2(matrix, tile_csr_ptr, rowA, colA, nnzA, csrRowPtrA, csrColIdxA, csrValA, BM, BN,
                  tile_count_temp, flag_g, valid_cols_g, col_to_rank_g, tile_count_g);
    exclusive_scan(matrix->tile_nnz, tilenum + 1);
    
    matrix->dnssize = (size_t)tilenum * BM * BN;
    matrix->Blockdense_Val = (MAT_VAL_TYPE *)malloc(matrix->dnssize * sizeof(MAT_VAL_TYPE));
    checkHostAlloc(matrix->Blockdense_Val, matrix->dnssize * sizeof(MAT_VAL_TYPE), "matrix->Blockdense_Val");
    
    convert_step3(matrix, tile_csr_ptr, nnz_temp, tile_count_temp, rowA, colA, nnzA, csrRowPtrA, csrColIdxA, csrValA, BM, BN,
                  flag_g, valid_cols_g, col_to_rank_g, tile_count_g);
    free(tile_csr_ptr);
}

// 先热身 20 次，再每 50 次检查频率；达到稳定条件或次数上限后结束
static void warmup_gpu(Tile_matrix *mat64, MAT_VAL_TYPE *d_Blockdense_Val_64, unsigned int *d_blkcoostylerowidx_64, int *d_blkcoostylerowidx_colstart_64, int *d_tile_colidx_64, cudaStream_t stream64,
                       Tile_matrix *mat16, MAT_VAL_TYPE *d_Blockdense_Val_16, unsigned int *d_blkcoostylerowidx_16, int *d_blkcoostylerowidx_colstart_16, int *d_tile_colidx_16, cudaStream_t stream16,
                       Tile_matrix *mat8, MAT_VAL_TYPE *d_Blockdense_Val_8, unsigned int *d_blkcoostylerowidx_8, int *d_blkcoostylerowidx_colstart_8, int *d_tile_colidx_8, cudaStream_t stream8,
                       int colB, MAT_VAL_TYPE *d_b, MAT_ANS_TYPE *d_c, cudaEvent_t stop, int reference_khz)
{
    // 无法查询频率时，保持固定 100 次热身
    int limit = reference_khz > 0 && get_gpu_clock() ? 1000 : 100;
    unsigned int previous = 0;
    for (int total = 0; total < limit;)
    {
        int count = total == 0 ? 20 : 50;
        if (count > limit - total) count = limit - total;
        for (int i = 0; i < count; i++)
        {
            if (mat64->tilenum > 0) switch_tilespmm_cuda_v64(mat64->rowblkblock, colB, d_Blockdense_Val_64, d_blkcoostylerowidx_64, d_blkcoostylerowidx_colstart_64, d_b, d_c, d_tile_colidx_64, stream64);
            if (mat16->tilenum > 0) switch_tilespmm_cuda_v16(mat16->rowblkblock, colB, d_Blockdense_Val_16, d_blkcoostylerowidx_16, d_blkcoostylerowidx_colstart_16, d_b, d_c, d_tile_colidx_16, stream16);
            if (mat8->tilenum > 0)  switch_tilespmm_cuda_v8(mat8->rowblkblock,  colB, d_Blockdense_Val_8,  d_blkcoostylerowidx_8,  d_blkcoostylerowidx_colstart_8,  d_b, d_c,  d_tile_colidx_8,  stream8);
        }
        checkCuda(cudaEventRecord(stop, cudaStreamLegacy), "cudaEventRecord");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
        total += count;
        unsigned int current = get_gpu_clock();
        if (reference_khz > 0 && previous && current && previous * 1000.0 >= reference_khz * 0.95 && current * 1000.0 >= reference_khz * 0.95 && fabsf((float)current / previous - 1.0f) <= 0.02f) break;
        previous = current;
    }
}

// 上传 Tile 数据，实测选择调度阈值，再执行正式计时和结果回传
void call_tilespmm_cuda(
                        Tile_matrix *mat64, int rowA_64, MAT_PTR_TYPE nnzA_64,
                        Tile_matrix *mat16, int rowA_16, MAT_PTR_TYPE nnzA_16,
                        Tile_matrix *mat8,  int rowA_8,  MAT_PTR_TYPE nnzA_8,
                        int colA, int colB, MAT_PTR_TYPE total_nnzA,
                        MAT_VAL_TYPE *b, MAT_ANS_TYPE *raw_c, const cudaDeviceProp& deviceProp,
                        double cpu_pre_time_ms, int logical_colB, FILE *fout)
{    
    int rowA = rowA_64 + rowA_16 + rowA_8;
    int row_padding = (rowA + 64) / 64 * 64; 
    MAT_VAL_TYPE *d_b;
    MAT_ANS_TYPE *d_c;
    
    checkCuda(cudaMalloc((void **)&d_b, (size_t)colA * (size_t)colB * sizeof(MAT_VAL_TYPE)), "cudaMalloc d_b");
    checkCuda(cudaMalloc((void **)&d_c, (size_t)row_padding * (size_t)colB * sizeof(MAT_ANS_TYPE)), "cudaMalloc d_c");
    checkCuda(cudaMemcpy(d_b, b, (size_t)colA * (size_t)colB * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice), "cudaMemcpy d_b");
    checkCuda(cudaMemset(d_c, 0, (size_t)row_padding * (size_t)colB * sizeof(MAT_ANS_TYPE)), "cudaMemset d_c");
    
    MAT_VAL_TYPE *d_Blockdense_Val_64 = NULL, *d_Blockdense_Val_16 = NULL, *d_Blockdense_Val_8 = NULL;
    unsigned int *d_blkcoostylerowidx_64 = NULL, *d_blkcoostylerowidx_16 = NULL, *d_blkcoostylerowidx_8 = NULL;
    int *d_blkcoostylerowidx_colstart_64 = NULL, *d_blkcoostylerowidx_colstart_16 = NULL, *d_blkcoostylerowidx_colstart_8 = NULL;
    int *d_tile_colidx_64 = NULL, *d_tile_colidx_16 = NULL, *d_tile_colidx_8 = NULL;

    if (mat64->tilenum > 0) 
    {
        int BN = mat64->dnssize / mat64->tilenum / 64;
        checkCuda(cudaMalloc((void **)&d_Blockdense_Val_64, mat64->dnssize * sizeof(MAT_VAL_TYPE)), "cudaMalloc d_Blockdense_Val_64");
        checkCuda(cudaMemcpy(d_Blockdense_Val_64, mat64->Blockdense_Val, mat64->dnssize * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice), "cudaMemcpy d_Blockdense_Val_64");
        checkCuda(cudaMalloc((void **)&d_tile_colidx_64, (size_t)mat64->tilenum * BN * sizeof(int)), "cudaMalloc d_tile_colidx_64");
        checkCuda(cudaMemcpy(d_tile_colidx_64, mat64->tile_colidx, (size_t)mat64->tilenum * BN * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_tile_colidx_64");
    }
    if (mat16->tilenum > 0) 
    {
        int BN = mat16->dnssize / mat16->tilenum / 16;
        checkCuda(cudaMalloc((void **)&d_Blockdense_Val_16, mat16->dnssize * sizeof(MAT_VAL_TYPE)), "cudaMalloc d_Blockdense_Val_16");
        checkCuda(cudaMemcpy(d_Blockdense_Val_16, mat16->Blockdense_Val, mat16->dnssize * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice), "cudaMemcpy d_Blockdense_Val_16");
        checkCuda(cudaMalloc((void **)&d_tile_colidx_16, (size_t)mat16->tilenum * BN * sizeof(int)), "cudaMalloc d_tile_colidx_16");
        checkCuda(cudaMemcpy(d_tile_colidx_16, mat16->tile_colidx, (size_t)mat16->tilenum * BN * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_tile_colidx_16");
    }
    if (mat8->tilenum > 0) 
    {
        int BN = mat8->dnssize / mat8->tilenum / 8;
        checkCuda(cudaMalloc((void **)&d_Blockdense_Val_8, mat8->dnssize * sizeof(MAT_VAL_TYPE)), "cudaMalloc d_Blockdense_Val_8");
        checkCuda(cudaMemcpy(d_Blockdense_Val_8, mat8->Blockdense_Val, mat8->dnssize * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice), "cudaMemcpy d_Blockdense_Val_8");
        checkCuda(cudaMalloc((void **)&d_tile_colidx_8, (size_t)mat8->tilenum * BN * sizeof(int)), "cudaMalloc d_tile_colidx_8");
        checkCuda(cudaMemcpy(d_tile_colidx_8, mat8->tile_colidx, (size_t)mat8->tilenum * BN * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_tile_colidx_8");
    }

    if (mat64->tilenum > 0) {
        int max_rowblk_64 = mat64->tilem + mat64->tilenum;
        checkCuda(cudaMalloc((void **)&d_blkcoostylerowidx_64, max_rowblk_64 * sizeof(unsigned int)), "cudaMalloc d_blkcoostylerowidx_64");
        checkCuda(cudaMalloc((void **)&d_blkcoostylerowidx_colstart_64, (max_rowblk_64 + 1) * sizeof(int)), "cudaMalloc d_blkcoostylerowidx_colstart_64");
    }
    if (mat16->tilenum > 0) {
        int max_rowblk_16 = mat16->tilem + mat16->tilenum;
        checkCuda(cudaMalloc((void **)&d_blkcoostylerowidx_16, max_rowblk_16 * sizeof(unsigned int)), "cudaMalloc d_blkcoostylerowidx_16");
        checkCuda(cudaMalloc((void **)&d_blkcoostylerowidx_colstart_16, (max_rowblk_16 + 1) * sizeof(int)), "cudaMalloc d_blkcoostylerowidx_colstart_16");
    }
    if (mat8->tilenum > 0) {
        int max_rowblk_8 = mat8->tilem + mat8->tilenum;
        checkCuda(cudaMalloc((void **)&d_blkcoostylerowidx_8, max_rowblk_8 * sizeof(unsigned int)), "cudaMalloc d_blkcoostylerowidx_8");  
        checkCuda(cudaMalloc((void **)&d_blkcoostylerowidx_colstart_8, (max_rowblk_8 + 1) * sizeof(int)), "cudaMalloc d_blkcoostylerowidx_colstart_8");
    }

    cudaStream_t stream64, stream16, stream8;
    checkCuda(cudaStreamCreate(&stream64), "cudaStreamCreate");
    checkCuda(cudaStreamCreate(&stream16), "cudaStreamCreate");
    checkCuda(cudaStreamCreate(&stream8), "cudaStreamCreate");
    
    cudaEvent_t start_pre, stop_pre;
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate");
    checkCuda(cudaEventCreate(&start_pre), "cudaEventCreate");
    checkCuda(cudaEventCreate(&stop_pre), "cudaEventCreate");

    int top4_64[4] = {32, 32, 32, 32};
    int top4_16[4] = {32, 32, 32, 32};
    int top4_8[4]  = {32, 32, 32, 32};

    checkCuda(cudaEventRecord(start_pre, cudaStreamLegacy), "cudaEventRecord");
#ifdef TFLOAT
    if (mat64->tilenum > 0) get_optimal_prefetch_th(64, colB, 1, mat64->tilem, mat64->tile_ptr, top4_64, deviceProp);
    if (mat16->tilenum > 0) get_optimal_prefetch_th(16, colB, 1, mat16->tilem, mat16->tile_ptr, top4_16, deviceProp);
    if (mat8->tilenum > 0)  get_optimal_prefetch_th(8,  colB, 1, mat8->tilem,  mat8->tile_ptr,  top4_8, deviceProp);
#else
    if (mat64->tilenum > 0) get_optimal_prefetch_th(64, colB, 0, mat64->tilem, mat64->tile_ptr, top4_64, deviceProp);
    if (mat16->tilenum > 0) get_optimal_prefetch_th(16, colB, 0, mat16->tilem, mat16->tile_ptr, top4_16, deviceProp);
    if (mat8->tilenum > 0)  get_optimal_prefetch_th(8,  colB, 0, mat8->tilem,  mat8->tile_ptr,  top4_8, deviceProp);
#endif

    int opts_64 = (mat64->tilenum > 0) ? 4 : 1;
    int opts_16 = (mat16->tilenum > 0) ? 4 : 1;
    int opts_8  = (mat8->tilenum > 0) ? 4 : 1;

    int unique_64 = 1, unique_16 = 1, unique_8 = 1;
    for (int i = 1; i < opts_64; i++) {
        int dup = 0;
        for (int j = 0; j < unique_64; j++) if (top4_64[i] == top4_64[j]) { dup = 1; break; }
        if (!dup) top4_64[unique_64++] = top4_64[i];
    }
    opts_64 = unique_64;

    for (int i = 1; i < opts_16; i++) {
        int dup = 0;
        for (int j = 0; j < unique_16; j++) if (top4_16[i] == top4_16[j]) { dup = 1; break; }
        if (!dup) top4_16[unique_16++] = top4_16[i];
    }
    opts_16 = unique_16;

    for (int i = 1; i < opts_8; i++) {
        int dup = 0;
        for (int j = 0; j < unique_8; j++) if (top4_8[i] == top4_8[j]) { dup = 1; break; }
        if (!dup) top4_8[unique_8++] = top4_8[i];
    }
    opts_8 = unique_8;

    float best_tune_time = 1e12f;
    int best_th64 = top4_64[0], best_th16 = top4_16[0], best_th8 = top4_8[0];
    checkCuda(cudaEventRecord(stop_pre, cudaStreamLegacy), "cudaEventRecord");
    checkCuda(cudaEventSynchronize(stop_pre), "cudaEventSynchronize");
    float test_ms1;
    checkCuda(cudaEventElapsedTime(&test_ms1, start_pre, stop_pre), "cudaEventElapsedTime");

    Prefetch_cache cache64 = {}, cache16 = {}, cache8 = {};

    // 热身开始
    if (mat64->tilenum > 0) update_metadata(mat64, best_th64, d_blkcoostylerowidx_64, d_blkcoostylerowidx_colstart_64, stream64, &cache64);
    if (mat16->tilenum > 0) update_metadata(mat16, best_th16, d_blkcoostylerowidx_16, d_blkcoostylerowidx_colstart_16, stream16, &cache16);
    if (mat8->tilenum > 0)  update_metadata(mat8,  best_th8,  d_blkcoostylerowidx_8,  d_blkcoostylerowidx_colstart_8,  stream8, &cache8);

    warmup_gpu(mat64, d_Blockdense_Val_64, d_blkcoostylerowidx_64, d_blkcoostylerowidx_colstart_64, d_tile_colidx_64, stream64,
               mat16, d_Blockdense_Val_16, d_blkcoostylerowidx_16, d_blkcoostylerowidx_colstart_16, d_tile_colidx_16, stream16,
               mat8, d_Blockdense_Val_8, d_blkcoostylerowidx_8, d_blkcoostylerowidx_colstart_8, d_tile_colidx_8, stream8,
               colB, d_b, d_c, stop, deviceProp.clockRate);
    // 热身结束

    
    checkCuda(cudaEventRecord(start_pre, cudaStreamLegacy), "cudaEventRecord");

    int total_runs = opts_64 * opts_16 * opts_8;
    if(total_runs > 1)
    {
        for (int i = 0; i < opts_64; i++) 
        {
            for (int j = 0; j < opts_16; j++) 
            {
                for (int k = 0; k < opts_8; k++) 
                {
                    
                    int cur_th64 = top4_64[i];
                    int cur_th16 = top4_16[j];
                    int cur_th8  = top4_8[k];

                    if (mat64->tilenum > 0) update_metadata(mat64, cur_th64, d_blkcoostylerowidx_64, d_blkcoostylerowidx_colstart_64, stream64, &cache64);
                    if (mat16->tilenum > 0) update_metadata(mat16, cur_th16, d_blkcoostylerowidx_16, d_blkcoostylerowidx_colstart_16, stream16, &cache16);
                    if (mat8->tilenum > 0)  update_metadata(mat8,  cur_th8,  d_blkcoostylerowidx_8,  d_blkcoostylerowidx_colstart_8,  stream8, &cache8);

                    for (int w = 0; w < 5; w++) 
                    {
                        if (mat64->tilenum > 0) switch_tilespmm_cuda_v64(mat64->rowblkblock, colB, d_Blockdense_Val_64, d_blkcoostylerowidx_64, d_blkcoostylerowidx_colstart_64, d_b, d_c, d_tile_colidx_64, stream64);
                        if (mat16->tilenum > 0) switch_tilespmm_cuda_v16(mat16->rowblkblock, colB, d_Blockdense_Val_16, d_blkcoostylerowidx_16, d_blkcoostylerowidx_colstart_16, d_b, d_c, d_tile_colidx_16, stream16);
                        if (mat8->tilenum > 0)  switch_tilespmm_cuda_v8(mat8->rowblkblock,  colB, d_Blockdense_Val_8,  d_blkcoostylerowidx_8,  d_blkcoostylerowidx_colstart_8,  d_b, d_c,  d_tile_colidx_8,  stream8);
                    }

                    checkCuda(cudaEventRecord(start, cudaStreamLegacy), "cudaEventRecord");
                    for (int iter = 0; iter < 10; iter++) 
                    {
                        if (mat64->tilenum > 0) switch_tilespmm_cuda_v64(mat64->rowblkblock, colB, d_Blockdense_Val_64, d_blkcoostylerowidx_64, d_blkcoostylerowidx_colstart_64, d_b, d_c, d_tile_colidx_64, stream64);
                        if (mat16->tilenum > 0) switch_tilespmm_cuda_v16(mat16->rowblkblock, colB, d_Blockdense_Val_16, d_blkcoostylerowidx_16, d_blkcoostylerowidx_colstart_16, d_b, d_c, d_tile_colidx_16, stream16);
                        if (mat8->tilenum > 0)  switch_tilespmm_cuda_v8(mat8->rowblkblock,  colB, d_Blockdense_Val_8,  d_blkcoostylerowidx_8,  d_blkcoostylerowidx_colstart_8,  d_b, d_c,  d_tile_colidx_8,  stream8);
                    }
                    checkCuda(cudaEventRecord(stop, cudaStreamLegacy), "cudaEventRecord");
                    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");

                    float ms = 0;
                    checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");

                    if (ms < best_tune_time) 
                    {
                        best_tune_time = ms;
                        best_th64 = cur_th64;
                        best_th16 = cur_th16;
                        best_th8  = cur_th8;
                    }
                }
            }
        }
    } 

    checkCuda(cudaEventRecord(stop_pre, cudaStreamLegacy), "cudaEventRecord");
    checkCuda(cudaEventSynchronize(stop_pre), "cudaEventSynchronize");
    float test_ms2;
    checkCuda(cudaEventElapsedTime(&test_ms2, start_pre, stop_pre), "cudaEventElapsedTime");
    if (mat64->tilenum > 0) update_metadata(mat64, best_th64, d_blkcoostylerowidx_64, d_blkcoostylerowidx_colstart_64, stream64, &cache64);
    if (mat16->tilenum > 0) update_metadata(mat16, best_th16, d_blkcoostylerowidx_16, d_blkcoostylerowidx_colstart_16, stream16, &cache16);
    if (mat8->tilenum > 0)  update_metadata(mat8,  best_th8,  d_blkcoostylerowidx_8,  d_blkcoostylerowidx_colstart_8,  stream8, &cache8);

    checkCuda(cudaMemset(d_c, 0, (size_t)row_padding * (size_t)colB * sizeof(MAT_ANS_TYPE)), "cudaMemset d_c");
    if (mat64->tilenum > 0) switch_tilespmm_cuda_v64(mat64->rowblkblock, colB, d_Blockdense_Val_64, d_blkcoostylerowidx_64, d_blkcoostylerowidx_colstart_64, d_b, d_c, d_tile_colidx_64, stream64);
    if (mat16->tilenum > 0) switch_tilespmm_cuda_v16(mat16->rowblkblock, colB, d_Blockdense_Val_16, d_blkcoostylerowidx_16, d_blkcoostylerowidx_colstart_16, d_b, d_c, d_tile_colidx_16, stream16);
    if (mat8->tilenum > 0)  switch_tilespmm_cuda_v8(mat8->rowblkblock,  colB, d_Blockdense_Val_8,  d_blkcoostylerowidx_8,  d_blkcoostylerowidx_colstart_8,  d_b, d_c,  d_tile_colidx_8,  stream8);
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    if (raw_c != NULL)
    {
        checkCuda(cudaMemcpy(raw_c, d_c, (size_t)rowA * (size_t)colB * sizeof(MAT_ANS_TYPE), cudaMemcpyDeviceToHost), "copy TileSpMM result");
    }

    warmup_gpu(mat64, d_Blockdense_Val_64, d_blkcoostylerowidx_64, d_blkcoostylerowidx_colstart_64, d_tile_colidx_64, stream64,
               mat16, d_Blockdense_Val_16, d_blkcoostylerowidx_16, d_blkcoostylerowidx_colstart_16, d_tile_colidx_16, stream16,
               mat8, d_Blockdense_Val_8, d_blkcoostylerowidx_8, d_blkcoostylerowidx_colstart_8, d_tile_colidx_8, stream8,
               colB, d_b, d_c, stop, deviceProp.clockRate);
    checkCuda(cudaMemset(d_c, 0, (size_t)row_padding * (size_t)colB * sizeof(MAT_ANS_TYPE)), "cudaMemset d_c");
    if (mat64->tilenum > 0) switch_tilespmm_cuda_v64(mat64->rowblkblock, colB, d_Blockdense_Val_64, d_blkcoostylerowidx_64, d_blkcoostylerowidx_colstart_64, d_b, d_c, d_tile_colidx_64, stream64);
    if (mat16->tilenum > 0) switch_tilespmm_cuda_v16(mat16->rowblkblock, colB, d_Blockdense_Val_16, d_blkcoostylerowidx_16, d_blkcoostylerowidx_colstart_16, d_b, d_c, d_tile_colidx_16, stream16);
    if (mat8->tilenum > 0)  switch_tilespmm_cuda_v8(mat8->rowblkblock,  colB, d_Blockdense_Val_8,  d_blkcoostylerowidx_8,  d_blkcoostylerowidx_colstart_8,  d_b, d_c,  d_tile_colidx_8,  stream8);

    checkCuda(cudaEventRecord(start, cudaStreamLegacy), "cudaEventRecord");
    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        if (mat64->tilenum > 0) switch_tilespmm_cuda_v64(mat64->rowblkblock, colB, d_Blockdense_Val_64, d_blkcoostylerowidx_64, d_blkcoostylerowidx_colstart_64, d_b, d_c, d_tile_colidx_64, stream64);
        if (mat16->tilenum > 0) switch_tilespmm_cuda_v16(mat16->rowblkblock, colB, d_Blockdense_Val_16, d_blkcoostylerowidx_16, d_blkcoostylerowidx_colstart_16, d_b, d_c, d_tile_colidx_16, stream16);
        if (mat8->tilenum > 0) switch_tilespmm_cuda_v8(mat8->rowblkblock,  colB, d_Blockdense_Val_8,  d_blkcoostylerowidx_8,  d_blkcoostylerowidx_colstart_8,  d_b, d_c,  d_tile_colidx_8,  stream8);
    }
    checkCuda(cudaEventRecord(stop, cudaStreamLegacy), "cudaEventRecord");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");

    float time_ms = 0;
    checkCuda(cudaEventElapsedTime(&time_ms, start, stop), "cudaEventElapsedTime");
    double time_cuda_spmm_base = time_ms / BENCH_REPEAT;

    double Tflops = (2 * (double)total_nnzA * 1.0e-6 / time_cuda_spmm_base * logical_colB) / 1000;
    printf("  TileSpMM runtime %4.3f ms, %4.3f TFlops\n", time_cuda_spmm_base, Tflops);

    // 预处理时间仍为原来的三段计时之和，单位为毫秒
    double pre_time_ms = cpu_pre_time_ms + (double)test_ms1 + (double)test_ms2;
    fprintf(fout, "%d,%d,%d,%.9g,%.9g,%.9g,", best_th64, best_th16, best_th8, pre_time_ms, time_cuda_spmm_base, Tflops);
    fflush(fout);

    free_prefetch_cache(mat64, &cache64);
    free_prefetch_cache(mat16, &cache16);
    free_prefetch_cache(mat8, &cache8);

    if (d_Blockdense_Val_64) checkCuda(cudaFree(d_Blockdense_Val_64), "cudaFree d_Blockdense_Val_64");
    if (d_blkcoostylerowidx_64) checkCuda(cudaFree(d_blkcoostylerowidx_64), "cudaFree d_blkcoostylerowidx_64");
    if (d_blkcoostylerowidx_colstart_64) checkCuda(cudaFree(d_blkcoostylerowidx_colstart_64), "cudaFree d_blkcoostylerowidx_colstart_64");
    if (d_tile_colidx_64) checkCuda(cudaFree(d_tile_colidx_64), "cudaFree d_tile_colidx_64");

    if (d_Blockdense_Val_16) checkCuda(cudaFree(d_Blockdense_Val_16), "cudaFree d_Blockdense_Val_16");
    if (d_blkcoostylerowidx_16) checkCuda(cudaFree(d_blkcoostylerowidx_16), "cudaFree d_blkcoostylerowidx_16");
    if (d_blkcoostylerowidx_colstart_16) checkCuda(cudaFree(d_blkcoostylerowidx_colstart_16), "cudaFree d_blkcoostylerowidx_colstart_16");
    if (d_tile_colidx_16) checkCuda(cudaFree(d_tile_colidx_16), "cudaFree d_tile_colidx_16");

    if (d_Blockdense_Val_8) checkCuda(cudaFree(d_Blockdense_Val_8), "cudaFree d_Blockdense_Val_8");
    if (d_blkcoostylerowidx_8) checkCuda(cudaFree(d_blkcoostylerowidx_8), "cudaFree d_blkcoostylerowidx_8");
    if (d_blkcoostylerowidx_colstart_8) checkCuda(cudaFree(d_blkcoostylerowidx_colstart_8), "cudaFree d_blkcoostylerowidx_colstart_8");
    if (d_tile_colidx_8) checkCuda(cudaFree(d_tile_colidx_8), "cudaFree d_tile_colidx_8");

    checkCuda(cudaFree(d_b), "cudaFree d_b");
    checkCuda(cudaFree(d_c), "cudaFree d_c");
    checkCuda(cudaStreamDestroy(stream64), "cudaStreamDestroy");
    checkCuda(cudaStreamDestroy(stream16), "cudaStreamDestroy");
    checkCuda(cudaStreamDestroy(stream8), "cudaStreamDestroy");
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy");
    checkCuda(cudaEventDestroy(start_pre), "cudaEventDestroy");
    checkCuda(cudaEventDestroy(stop_pre), "cudaEventDestroy");
}
