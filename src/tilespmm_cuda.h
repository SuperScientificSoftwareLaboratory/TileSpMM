#ifndef TILESPMM_CUDA_H
#define TILESPMM_CUDA_H

#include "common.h"
#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
using namespace nvcuda;

// 每个block64线程，每个warp处理矩阵B的32列
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N>
__global__ void stir_spmm_cuda_kernel_v8(int colB,
                                            const MAT_VAL_TYPE* __restrict__ d_Blockdense_Val,
                                            int rowblkblock,
                                            const unsigned int* __restrict__ d_blkcoostylerowidx,
                                            const int* __restrict__ d_blkcoostylerowidx_colstart,
                                            const MAT_VAL_TYPE* __restrict__ d_b,
                                            MAT_ANS_TYPE* __restrict__ d_c,  
                                            const int* __restrict__ d_tile_colidx)
{
    const int blk_idx = blockIdx.z * gridDim.y + blockIdx.y;
    if (blk_idx >= rowblkblock) return;

    const int thread_id = threadIdx.x;
    const int lane_id = thread_id & 31; 
    const int dimN_index = blockIdx.x * STRIDE_N;
    const int tile_start = (thread_id >> 5) * Tile_N; 

    __shared__ MAT_VAL_TYPE s_dense_tile[BLOCK_SIZE_N * stride_padded_half];
    float reg_c[2][4] = {}; 

    size_t rowblkjstart = __ldg(&d_blkcoostylerowidx_colstart[blk_idx]);
    int num_tiles = __ldg(&d_blkcoostylerowidx_colstart[blk_idx + 1]) - rowblkjstart;
    
    if (num_tiles == 0) return;
    size_t denseoffset = rowblkjstart * 128;

    for (int i = 0; i < num_tiles; i++)
    {
        size_t current_tile_idx = rowblkjstart + i;
#pragma unroll
        for (int k = 0; k < 2; k++) 
        {
            int v_idx = thread_id + k * 64; 
            int original_row = __ldg(&d_tile_colidx[current_tile_idx * BLOCK_SIZE_N + (v_idx >> 3)]);
            *((int4*)&s_dense_tile[(v_idx >> 3) * stride_padded_half + (v_idx & 7) * 8]) = 
                *((const int4*)&d_b[(long long)original_row * colB + dimN_index + ((v_idx & 7) << 3)]);
        }

        unsigned reg_b_frag[4]; 
        int2 loaded_val = *((int2 *)(&d_Blockdense_Val[denseoffset]) + lane_id);
        __syncthreads(); 
        int base_smem_offset = (lane_id % 16) * stride_padded_half + tile_start + ((lane_id >> 4) * 8);
        
        ldmatrix_m16n16_x4_trans(reg_b_frag, &s_dense_tile[base_smem_offset]);
        mma_m16n8k16_f32(reg_c[0], reg_b_frag, (unsigned*)&loaded_val, reg_c[0]);

        ldmatrix_m16n16_x4_trans(reg_b_frag, &s_dense_tile[base_smem_offset + 16]);
        mma_m16n8k16_f32(reg_c[1], reg_b_frag, (unsigned*)&loaded_val, reg_c[1]);

        __syncthreads(); 
        denseoffset += 128;

    } // End Main Loop

    float* s_out_tile = (float*)s_dense_tile; 
    
    store_transposed_mma_m16n8k16(reg_c[0], s_out_tile + tile_start, stride_padded_float);
    store_transposed_mma_m16n8k16(reg_c[1], s_out_tile + tile_start + 16, stride_padded_float);
    __syncthreads(); 
    int coostyleblkrowidx = __ldg(&d_blkcoostylerowidx[blk_idx]);
    int blki = coostyleblkrowidx & 0x7FFFFFFF;
    bool signbit = (coostyleblkrowidx >> 31) & 0x1;
    long long base_addr_start = (long long)blki * colB + dimN_index + thread_id;
    
    if (signbit) 
    { 
        #pragma unroll
        for (int i = 0; i < 8; i++) 
        {
            atomicAdd(&d_c[base_addr_start], s_out_tile[i * stride_padded_float + thread_id]);
            base_addr_start += colB;
        }
    } 
    else 
    { 
        MAT_ANS_TYPE *dst_ptr = &d_c[base_addr_start];
        #pragma unroll
        for (int i = 0; i < 8; i++) 
        {
            float val = s_out_tile[i * stride_padded_float + thread_id];
            asm volatile("st.global.cs.f32 [%0], %1;" ::"l"(dst_ptr), "f"(val));
            dst_ptr += colB;
        }
    }
}

// 配置：1 Block = 4 Warps (128 threads)
// 每个 Warp 处理一个 rowblkblock (8行 A) x 32列 B
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N>
__global__ void stir_spmm_cuda_kernel_v8_32(int colB,
                                        const MAT_VAL_TYPE* __restrict__ d_Blockdense_Val,
                                        int rowblkblock,
                                        const unsigned int* __restrict__ d_blkcoostylerowidx,
                                        const int* __restrict__ d_blkcoostylerowidx_colstart,
                                        const MAT_VAL_TYPE* __restrict__ d_b,
                                        MAT_ANS_TYPE* __restrict__ d_c,  
                                        const int* __restrict__ d_tile_colidx)
{
    const int warp_id = threadIdx.x >> 5; 
    const int lane_id = threadIdx.x & 31;
    
    // const int blk_idx = blockIdx.x * 4 + warp_id;
    const int blk_idx = (blockIdx.z * gridDim.y + blockIdx.y) * 4 + warp_id;

    if (blk_idx >= rowblkblock) return;
    const int stride_32 = 32 + PADDING_FLOAT;
    const int SMEM_TILE_STEP = BLOCK_SIZE_N * stride_32;
    __shared__ MAT_VAL_TYPE s_smem_base[4 * BLOCK_SIZE_N * stride_32];
    MAT_VAL_TYPE* s_dense_tile = s_smem_base + warp_id * (BLOCK_SIZE_N * stride_32);
    float reg_c[2][4] = {}; 

    size_t rowblkjstart = __ldg(&d_blkcoostylerowidx_colstart[blk_idx]);
    int num_tiles = __ldg(&d_blkcoostylerowidx_colstart[blk_idx + 1]) - rowblkjstart;
    
    if (num_tiles == 0) return;
    size_t denseoffset = rowblkjstart * 128; 

    for (int i = 0; i < num_tiles; i++)
    {
        size_t current_tile_idx = rowblkjstart + i;

        #pragma unroll
        for(int k=0; k<2; k++) {
            int task_id = lane_id + k * 32; 
            int row_b = task_id >> 2; 
            int col_chunk = task_id & 3; 
            int original_row = __ldg(&d_tile_colidx[current_tile_idx * BLOCK_SIZE_N + row_b]);

            *((int4*)&s_dense_tile[row_b * stride_32 + col_chunk * 8]) = 
                *((const int4*)&d_b[(long long)original_row * 32 + (col_chunk << 3)]);
        }

        int2 loaded_val = *((int2 *)(&d_Blockdense_Val[denseoffset]) + lane_id);
        
        __syncthreads(); 
        
        unsigned reg_b_frag[4]; 

        int base_smem_offset = (lane_id % 16) * stride_32 + ((lane_id >> 4) * 8);
        ldmatrix_m16n16_x4_trans(reg_b_frag, &s_dense_tile[base_smem_offset]);
        mma_m16n8k16_f32(reg_c[0], reg_b_frag, (unsigned*)&loaded_val, reg_c[0]);

        ldmatrix_m16n16_x4_trans(reg_b_frag, &s_dense_tile[base_smem_offset + 16]);
        mma_m16n8k16_f32(reg_c[1], reg_b_frag, (unsigned*)&loaded_val, reg_c[1]);

        __syncthreads(); 
        denseoffset += 128;
    } 
    
    float* s_out_tile = (float*)s_dense_tile; 
    
    store_transposed_mma_m16n8k16(reg_c[0], s_out_tile, stride_32);
    store_transposed_mma_m16n8k16(reg_c[1], s_out_tile + 16, stride_32);
    
    __syncthreads();

    int coostyleblkrowidx = __ldg(&d_blkcoostylerowidx[blk_idx]);
    int blki = coostyleblkrowidx & 0x7FFFFFFF;
    bool signbit = (coostyleblkrowidx >> 31) & 0x1;

    long long base_addr_start = (long long)blki * colB + lane_id;
    
    if (signbit) 
    { 
        #pragma unroll
        for (int row = 0; row < 8; row++) 
        {
            float val = s_out_tile[row * stride_32 + lane_id];
            atomicAdd(&d_c[base_addr_start], val);
            base_addr_start += colB; // 下一行
        }
    } 
    else 
    { 
        MAT_ANS_TYPE *dst_ptr = &d_c[base_addr_start];
        #pragma unroll
        for (int row = 0; row < 8; row++) 
        {
            float val = s_out_tile[row * stride_32 + lane_id];
            asm volatile("st.global.cs.f32 [%0], %1;" ::"l"(dst_ptr), "f"(val));
            dst_ptr += colB;
        }
    }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N>
__global__ void stir_spmm_cuda_kernel_v8_tf32(int colB,
                                              const MAT_VAL_TYPE* __restrict__ d_Blockdense_Val,
                                              int rowblkblock,
                                              const unsigned int* __restrict__ d_blkcoostylerowidx,
                                              const int* __restrict__ d_blkcoostylerowidx_colstart,
                                              const MAT_VAL_TYPE* __restrict__ d_b,
                                              MAT_ANS_TYPE* __restrict__ d_c,  
                                              const int* __restrict__ d_tile_colidx)
{
    const int blk_idx = blockIdx.z * gridDim.y + blockIdx.y;
    if (blk_idx >= rowblkblock) return;

    const int thread_id = threadIdx.x;
    const int lane_id = thread_id & 31; 
    const int dimN_index = blockIdx.x * STRIDE_N;
    const int tile_start = (thread_id >> 5) * Tile_N; 

    __shared__ float s_dense_tile[BLOCK_SIZE_N * stride_padded_float];

    float reg_c[2][4] = {0.0f}; 

    size_t rowblkjstart = __ldg(&d_blkcoostylerowidx_colstart[blk_idx]);
    int num_tiles = __ldg(&d_blkcoostylerowidx_colstart[blk_idx + 1]) - rowblkjstart;
    
    if (num_tiles == 0) return;
    
    size_t denseoffset = rowblkjstart * 64; 

    for (int i = 0; i < num_tiles; i++)
    {
        int current_tile_idx = rowblkjstart + i;
        #pragma unroll
        for (int k = 0; k < 2; k++) 
        {

            int v_idx = thread_id + k * 64; 
            int row_idx = v_idx >> 4;
            int original_row = __ldg(&d_tile_colidx[current_tile_idx * BLOCK_SIZE_N + row_idx]);
            int col_offset = (v_idx & 15) << 2;

            *((float4*)&s_dense_tile[row_idx * stride_padded_float + col_offset]) = 
                *((const float4*)&d_b[(long long)original_row * colB + dimN_index + col_offset]);
        }

        uint2 val = reinterpret_cast<const uint2*>(d_Blockdense_Val + denseoffset)[lane_id];

        __syncthreads(); 
        
        unsigned frag_b[4];

        int row_0 = lane_id & 3; 
        int row_1 = row_0 + 4;

        int col_off = lane_id >> 2; 
        int warp_col_base = (thread_id >> 5) * 32; 
        
        int c0 = warp_col_base + col_off;     
        int c1 = c0 + 8;                

        frag_b[0] = *((unsigned*)&s_dense_tile[row_0 * stride_padded_float + c0]);
        frag_b[1] = *((unsigned*)&s_dense_tile[row_0 * stride_padded_float + c1]);
        frag_b[2] = *((unsigned*)&s_dense_tile[row_1 * stride_padded_float + c0]);
        frag_b[3] = *((unsigned*)&s_dense_tile[row_1 * stride_padded_float + c1]);

        mma_m16n8k8_tf32(reg_c[0], frag_b, (unsigned*)&val, reg_c[0]);

        int warp_col_base_2 = warp_col_base + 16;
        
        c0 = warp_col_base_2 + col_off;
        c1 = c0 + 8;

        frag_b[0] = *((unsigned*)&s_dense_tile[row_0 * stride_padded_float + c0]);
        frag_b[1] = *((unsigned*)&s_dense_tile[row_0 * stride_padded_float + c1]);
        frag_b[2] = *((unsigned*)&s_dense_tile[row_1 * stride_padded_float + c0]);
        frag_b[3] = *((unsigned*)&s_dense_tile[row_1 * stride_padded_float + c1]);

        mma_m16n8k8_tf32(reg_c[1], frag_b, (unsigned*)&val, reg_c[1]);

        __syncthreads(); 
        denseoffset += 64; 
    }

    float* s_out_tile = (float*)s_dense_tile; 
    
    store_transposed_mma_m16n8k16(reg_c[0], s_out_tile + tile_start, stride_padded_float);
    store_transposed_mma_m16n8k16(reg_c[1], s_out_tile + tile_start + 16, stride_padded_float);
    __syncthreads(); 
    int coostyleblkrowidx = __ldg(&d_blkcoostylerowidx[blk_idx]);
    int blki = coostyleblkrowidx & 0x7FFFFFFF;
    bool signbit = (coostyleblkrowidx >> 31) & 0x1;
    long long base_addr_start = (long long)blki * colB + dimN_index + thread_id;
    
    if (signbit) 
    { 
        #pragma unroll
        for (int i = 0; i < 8; i++) 
        {
            atomicAdd(&d_c[base_addr_start], s_out_tile[i * stride_padded_float + thread_id]);
            base_addr_start += colB;
        }
    } 
    else 
    { 
        MAT_ANS_TYPE *dst_ptr = &d_c[base_addr_start];
        #pragma unroll
        for (int i = 0; i < 8; i++) 
        {
            float val = s_out_tile[i * stride_padded_float + thread_id];
            asm volatile("st.global.cs.f32 [%0], %1;" ::"l"(dst_ptr), "f"(val));
            dst_ptr += colB;
        }
    }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N>
__global__ void stir_spmm_cuda_kernel_v8_tf32_32(int colB,
                                                  const MAT_VAL_TYPE* __restrict__ d_Blockdense_Val,
                                                  int rowblkblock,
                                                  const unsigned int* __restrict__ d_blkcoostylerowidx,
                                                  const int* __restrict__ d_blkcoostylerowidx_colstart,
                                                  const MAT_VAL_TYPE* __restrict__ d_b,
                                                  MAT_ANS_TYPE* __restrict__ d_c,  
                                                  const int* __restrict__ d_tile_colidx)
{
    const int warp_id = threadIdx.x >> 5; 
    const int lane_id = threadIdx.x & 31;
    // const int blk_idx = blockIdx.x * 4 + warp_id;
    const int blk_idx = (blockIdx.z * gridDim.y + blockIdx.y) * 4 + warp_id;

    if (blk_idx >= rowblkblock) return;

    const int stride_32 = Tile_N + PADDING_FLOAT;

    __shared__ float s_smem_base[4 * BLOCK_SIZE_N * stride_32];
    float* s_dense_tile = s_smem_base + warp_id * (BLOCK_SIZE_N * stride_32);

    float reg_c[2][4] = {0.0f}; 

    size_t rowblkjstart = __ldg(&d_blkcoostylerowidx_colstart[blk_idx]);
    int num_tiles = __ldg(&d_blkcoostylerowidx_colstart[blk_idx + 1]) - rowblkjstart;
    
    if (num_tiles == 0) return;
    size_t denseoffset = rowblkjstart * 64; 

    for (int i = 0; i < num_tiles; i++)
    {
        int current_tile_idx = rowblkjstart + i;
        #pragma unroll
        for(int k = 0; k < 2; k++) {
            int task_id = lane_id + k * 32; 
            int row_b = task_id >> 3; 
            int col_chunk = task_id & 7; 
            int col_offset = col_chunk << 2;
            int original_row = __ldg(&d_tile_colidx[current_tile_idx * BLOCK_SIZE_N + row_b]);

            *((float4*)&s_dense_tile[row_b * stride_32 + col_offset]) = 
                *((const float4*)&d_b[(long long)original_row * colB + col_offset]);
        }

        uint2 val = reinterpret_cast<const uint2*>(d_Blockdense_Val + denseoffset)[lane_id];
        __syncwarp(); 
        unsigned frag_b[4]; 

        int row_0 = lane_id & 3; 
        int row_1 = row_0 + 4;
        int col_off = lane_id >> 2; 

        int c0 = col_off; 
        int c1 = c0 + 8;

        frag_b[0] = *((unsigned*)&s_dense_tile[row_0 * stride_32 + c0]);
        frag_b[1] = *((unsigned*)&s_dense_tile[row_0 * stride_32 + c1]);
        frag_b[2] = *((unsigned*)&s_dense_tile[row_1 * stride_32 + c0]);
        frag_b[3] = *((unsigned*)&s_dense_tile[row_1 * stride_32 + c1]);

        mma_m16n8k8_tf32(reg_c[0], frag_b, (unsigned*)&val, reg_c[0]);

        c0 = col_off + 16;
        c1 = c0 + 8;

        frag_b[0] = *((unsigned*)&s_dense_tile[row_0 * stride_32 + c0]);
        frag_b[1] = *((unsigned*)&s_dense_tile[row_0 * stride_32 + c1]);
        frag_b[2] = *((unsigned*)&s_dense_tile[row_1 * stride_32 + c0]);
        frag_b[3] = *((unsigned*)&s_dense_tile[row_1 * stride_32 + c1]);

        mma_m16n8k8_tf32(reg_c[1], frag_b, (unsigned*)&val, reg_c[1]);

        __syncwarp(); 
        denseoffset += 64;
    } 

    float* s_out_tile = (float*)s_dense_tile; 

    store_transposed_mma_m16n8k16(reg_c[0], s_out_tile, stride_32);
    store_transposed_mma_m16n8k16(reg_c[1], s_out_tile + 16, stride_32);
    
    __syncwarp();

    int coostyleblkrowidx = __ldg(&d_blkcoostylerowidx[blk_idx]);
    int blki = coostyleblkrowidx & 0x7FFFFFFF;
    bool signbit = (coostyleblkrowidx >> 31) & 0x1;

    long long base_addr_start = (long long)blki * colB + lane_id;
    
    if (signbit) 
    { 
        #pragma unroll
        for (int row = 0; row < 8; row++) 
        {
            float val = s_out_tile[row * stride_32 + lane_id];
            atomicAdd(&d_c[base_addr_start], val);
            base_addr_start += colB; 
        }
    } 
    else 
    { 
        MAT_ANS_TYPE *dst_ptr = &d_c[base_addr_start];
        #pragma unroll
        for (int row = 0; row < 8; row++) 
        {
            float val = s_out_tile[row * stride_32 + lane_id];
            asm volatile("st.global.cs.f32 [%0], %1;" ::"l"(dst_ptr), "f"(val));
            dst_ptr += colB; 
        }
    }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N>
__global__ void stir_spmm_cuda_kernel_v16_tf32(const int colB,
                                            const float* __restrict__ d_Blockdense_Val,
                                            const int rowblkblock,
                                            const unsigned int* __restrict__ d_blkcoostylerowidx,
                                            const int* __restrict__ d_blkcoostylerowidx_colstart,
                                            const float* __restrict__ d_b,
                                            float* __restrict__ d_c,
                                            const int* __restrict__ d_tile_colidx)
{
    const int blk_idx = blockIdx.z * gridDim.y + blockIdx.y;
    
    if (blk_idx >= rowblkblock) return;
    const int thread_id = threadIdx.x;
    const int dimN_index = blockIdx.x * STRIDE_N;
    const int lane_id = thread_id & 31; 
    const int SMEM_TILE_STEP = BLOCK_SIZE_N * (STRIDE_N + PADDING_FLOAT);

    __shared__ float s_dense_tile[2 * BLOCK_SIZE_N * (STRIDE_N + PADDING_FLOAT)];

    const int coostyleblkrowidx = d_blkcoostylerowidx[blk_idx];
    const int signbit = (coostyleblkrowidx >> 31) & 0x1;
    const int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;
    const size_t rowblkjstart = __ldg(&d_blkcoostylerowidx_colstart[blk_idx]);
    const int num_tiles = __ldg(&d_blkcoostylerowidx_colstart[blk_idx + 1]) - rowblkjstart;

    if (num_tiles == 0) return;

    const float4* ptr_A = (const float4*)d_Blockdense_Val + rowblkjstart * 32;
    const int* ptr_col_idx = d_tile_colidx + rowblkjstart * BLOCK_SIZE_N;
    
    const int rowid = thread_id >> 4;
    const int colid = (thread_id & 15) << 2;

    const int smem_offset_0 = rowid * (STRIDE_N + PADDING_FLOAT) + colid;
    const int smem_offset_1 = smem_offset_0 + 4 * (STRIDE_N + PADDING_FLOAT);
    const long long col_offset = dimN_index + colid;

    unsigned reg_b[4][2];
    float reg_c[4][4] = {0.0f};

    int prefetch_indices[2];
    float* buffer_ptr = s_dense_tile;

    int idx_0 = __ldg(ptr_col_idx + rowid); 
    int idx_1 = __ldg(ptr_col_idx + rowid + 4); 

    long long glob_b_offset = (long long)idx_0 * colB + col_offset;
    cp_async_int4(&buffer_ptr[smem_offset_0], &d_b[glob_b_offset]);

    long long glob_b_offset_1 = (long long)idx_1 * colB + col_offset;
    cp_async_int4(&buffer_ptr[smem_offset_1], &d_b[glob_b_offset_1]);

    if (num_tiles > 1) {
        prefetch_indices[0] = __ldg(ptr_col_idx + BLOCK_SIZE_N + rowid); 
        prefetch_indices[1] = __ldg(ptr_col_idx + BLOCK_SIZE_N + rowid + 4);
    }
    
    cp_async_commit();

    for (int i = 0; i < num_tiles; i++)
    {
        if (i < num_tiles - 1)
        {
            float* smem_write_ptr = s_dense_tile + ((i + 1) & 1) * SMEM_TILE_STEP;
            
            long long next_glob_b_0 = (long long)prefetch_indices[0] * colB + col_offset;
            cp_async_int4(&smem_write_ptr[smem_offset_0], &d_b[next_glob_b_0]);

            long long next_glob_b_1 = (long long)prefetch_indices[1] * colB + col_offset;
            cp_async_int4(&smem_write_ptr[smem_offset_1], &d_b[next_glob_b_1]);
            
            cp_async_commit();

            if (i < num_tiles - 2) {
                int offset = (i + 2) * BLOCK_SIZE_N; 
                prefetch_indices[0] = __ldg(ptr_col_idx + offset + rowid);
                prefetch_indices[1] = __ldg(ptr_col_idx + offset + rowid + 4);
            }
        }

        if (i < num_tiles - 1) cp_async_wait_group<1>();
        else cp_async_wait_group<0>();
        
        __syncthreads();

        {
            float* tile_ptr = s_dense_tile + (i & 1) * SMEM_TILE_STEP;
            unsigned *smem_u = (unsigned*)tile_ptr;

            float4 loaded_val = ptr_A[lane_id];
            unsigned *reg_a_ptr = (unsigned*)(&loaded_val);
            
            int smem_stride = STRIDE_N + PADDING_FLOAT;
            int tile_start_lane = (thread_id >> 5) * Tile_N;
            int local_row = lane_id & 3;
            int local_col = lane_id >> 2;
            int row_base = local_row * smem_stride + local_col;
            int smem_idx0 = row_base + tile_start_lane;
            int smem_idx1 = smem_idx0 + 4 * smem_stride;

            reg_b[0][0] = smem_u[smem_idx0];
            reg_b[0][1] = smem_u[smem_idx1];
            smem_idx0 += 8;
            smem_idx1 += 8;
            reg_b[1][0] = smem_u[smem_idx0];
            reg_b[1][1] = smem_u[smem_idx1];
            smem_idx0 += 8;
            smem_idx1 += 8;
            reg_b[2][0] = smem_u[smem_idx0];
            reg_b[2][1] = smem_u[smem_idx1];
            smem_idx0 += 8;
            smem_idx1 += 8;
            reg_b[3][0] = smem_u[smem_idx0];
            reg_b[3][1] = smem_u[smem_idx1];

            mma_m16n8k8_tf32(reg_c[0], reg_a_ptr, reg_b[0], reg_c[0]);
            mma_m16n8k8_tf32(reg_c[1], reg_a_ptr, reg_b[1], reg_c[1]);
            mma_m16n8k8_tf32(reg_c[2], reg_a_ptr, reg_b[2], reg_c[2]);
            mma_m16n8k8_tf32(reg_c[3], reg_a_ptr, reg_b[3], reg_c[3]);
        }
        
        ptr_A += 32; 
        __syncthreads();
    }

    int group_id = lane_id >> 2;   // 0-7
    int tile_start_write = (thread_id >> 5) * Tile_N;
    long long row0_addr = (long long)(blki + group_id) * colB + dimN_index;
    long long row1_addr = row0_addr + 8 * colB;
    
    int col_base_start = tile_start_write + (lane_id & 3) * 2;

    if (signbit) 
    {
        #pragma unroll
        for (int k = 0; k < 4; k++) 
        {
            int current_col = col_base_start + (k << 3); 
            
            atomicAdd(&d_c[row0_addr + current_col], reg_c[k][0]);
            atomicAdd(&d_c[row0_addr + current_col + 1], reg_c[k][1]);
            atomicAdd(&d_c[row1_addr + current_col], reg_c[k][2]);
            atomicAdd(&d_c[row1_addr + current_col + 1], reg_c[k][3]);
        }
    } 
    else
    {
        #pragma unroll
        for (int k = 0; k < 4; k++) 
        {
            int current_col = col_base_start + (k << 3);
            
            float2* val2 = reinterpret_cast<float2*>(&reg_c[k][0]);
            *(float2*)(&d_c[row0_addr + current_col]) = val2[0];
            *(float2*)(&d_c[row1_addr + current_col]) = val2[1];
        }
    }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N>
__global__ void stir_spmm_cuda_kernel_v16_tf32_32(const int colB, 
                                            const float* __restrict__ d_Blockdense_Val,
                                            const int rowblkblock,
                                            const unsigned int* __restrict__ d_blkcoostylerowidx,
                                            const int* __restrict__ d_blkcoostylerowidx_colstart,
                                            const float* __restrict__ d_b,
                                            float* __restrict__ d_c,
                                            const int* __restrict__ d_tile_colidx)
{
    const int warp_id = threadIdx.x >> 5; 
    const int lane_id = threadIdx.x & 31; 
    
    // const int blk_idx = blockIdx.x * 4 + warp_id;
    const int blk_idx = (blockIdx.z * gridDim.y + blockIdx.y) * 4 + warp_id;
    
    if (blk_idx >= rowblkblock) return;

    const int stride_32 = 32 + PADDING_FLOAT;
    const int SMEM_TILE_STEP = BLOCK_SIZE_N * stride_32;
    __shared__ float s_dense_tile[4][2 * SMEM_TILE_STEP]; 

    float* warp_smem = s_dense_tile[warp_id];

    const int coostyleblkrowidx = d_blkcoostylerowidx[blk_idx];
    const int signbit = (coostyleblkrowidx >> 31) & 0x1;
    const int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;
    const size_t rowblkjstart = __ldg(&d_blkcoostylerowidx_colstart[blk_idx]);
    const int num_tiles = __ldg(&d_blkcoostylerowidx_colstart[blk_idx + 1]) - rowblkjstart;

    if (num_tiles == 0) return;

    const float4* ptr_A = (const float4*)d_Blockdense_Val + rowblkjstart * 32;
    const int* ptr_col_idx = d_tile_colidx + rowblkjstart * BLOCK_SIZE_N;

    const int rowid = lane_id >> 2;         
    const int colid = (lane_id & 3) << 3;  
    const int smem_offset = rowid * stride_32 + colid;

    int idx = __ldg(ptr_col_idx + rowid); 
    long long glob_b_offset = (long long)idx * 32 + colid;

    cp_async_int4(&warp_smem[smem_offset], &d_b[glob_b_offset]);
    cp_async_int4(&warp_smem[smem_offset + 4], &d_b[glob_b_offset + 4]);

    int prefetch_index = 0;
    if (num_tiles > 1) {
        prefetch_index = __ldg(ptr_col_idx + BLOCK_SIZE_N + rowid); 
    }
    
    cp_async_commit();

    unsigned reg_b[4][2];
    float reg_c[4][4] = {0.0f};

    for (int i = 0; i < num_tiles; i++)
    {
        if (i < num_tiles - 1)
        {
            float* smem_write_ptr = warp_smem + ((i + 1) & 1) * SMEM_TILE_STEP;
            long long next_glob_b = (long long)prefetch_index * 32 + colid;

            cp_async_int4(&smem_write_ptr[smem_offset], &d_b[next_glob_b]);
            cp_async_int4(&smem_write_ptr[smem_offset + 4], &d_b[next_glob_b + 4]);
            
            cp_async_commit();

            if (i < num_tiles - 2) {
                int offset = (i + 2) * BLOCK_SIZE_N; 
                prefetch_index = __ldg(ptr_col_idx + offset + rowid);
            }
        }

        if (i < num_tiles - 1) cp_async_wait_group<1>();
        else cp_async_wait_group<0>();

        __syncwarp(); 

        {
            float* tile_ptr = warp_smem + (i & 1) * SMEM_TILE_STEP;
            unsigned *smem_u = (unsigned*)tile_ptr;

            float4 loaded_val = ptr_A[lane_id];
            unsigned *reg_a_ptr = (unsigned*)(&loaded_val);

            int local_row = lane_id & 3;
            int local_col = lane_id >> 2;
            int row_base = local_row * stride_32 + local_col;
            
            int smem_idx0 = row_base;
            int smem_idx1 = smem_idx0 + 4 * stride_32;

            reg_b[0][0] = smem_u[smem_idx0];
            reg_b[0][1] = smem_u[smem_idx1];
            smem_idx0 += 8;
            smem_idx1 += 8;
            reg_b[1][0] = smem_u[smem_idx0];
            reg_b[1][1] = smem_u[smem_idx1];
            smem_idx0 += 8;
            smem_idx1 += 8;
            reg_b[2][0] = smem_u[smem_idx0];
            reg_b[2][1] = smem_u[smem_idx1];
            smem_idx0 += 8;
            smem_idx1 += 8;
            reg_b[3][0] = smem_u[smem_idx0];
            reg_b[3][1] = smem_u[smem_idx1];

            mma_m16n8k8_tf32(reg_c[0], reg_a_ptr, reg_b[0], reg_c[0]);
            mma_m16n8k8_tf32(reg_c[1], reg_a_ptr, reg_b[1], reg_c[1]);
            mma_m16n8k8_tf32(reg_c[2], reg_a_ptr, reg_b[2], reg_c[2]);
            mma_m16n8k8_tf32(reg_c[3], reg_a_ptr, reg_b[3], reg_c[3]);
        }
        
        ptr_A += 32; 
        
        __syncwarp();
    }

    int group_id = lane_id >> 2; 
    long long row0_addr = (long long)(blki + group_id) * 32;
    long long row1_addr = row0_addr + 8 * 32;
    
    int col_base_start = (lane_id & 3) * 2;

    if (signbit) 
    {
        #pragma unroll
        for (int k = 0; k < 4; k++) 
        {
            int current_col = col_base_start + (k << 3); 
            
            atomicAdd(&d_c[row0_addr + current_col], reg_c[k][0]);
            atomicAdd(&d_c[row0_addr + current_col + 1], reg_c[k][1]);
            atomicAdd(&d_c[row1_addr + current_col], reg_c[k][2]);
            atomicAdd(&d_c[row1_addr + current_col + 1], reg_c[k][3]);
        }
    } 
    else
    {
        #pragma unroll
        for (int k = 0; k < 4; k++) 
        {
            int current_col = col_base_start + (k << 3);
            
            float2* val2 = reinterpret_cast<float2*>(&reg_c[k][0]);
            *(float2*)(&d_c[row0_addr + current_col]) = val2[0];
            *(float2*)(&d_c[row1_addr + current_col]) = val2[1];
        }
    }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N>
__global__ void stir_spmm_cuda_kernel_v16(const int colB,
                                            const MAT_VAL_TYPE* __restrict__ d_Blockdense_Val,
                                            const int rowblkblock,
                                            const unsigned int* __restrict__ d_blkcoostylerowidx,
                                            const int* __restrict__ d_blkcoostylerowidx_colstart,
                                            const MAT_VAL_TYPE* __restrict__ d_b,
                                            MAT_ANS_TYPE* __restrict__ d_c,
                                            const int* __restrict__ d_tile_colidx)
{
    const int blk_idx = blockIdx.z * gridDim.y + blockIdx.y;
    
    if (blk_idx >= rowblkblock) return;

    const int thread_id = threadIdx.x;
    const int dimN_index = blockIdx.x * STRIDE_N;
    const int lane_id = thread_id & 31; 
    
    const int SMEM_TILE_STEP = BLOCK_SIZE_N * (STRIDE_N + PADDING_HALF);
    __shared__ MAT_VAL_TYPE s_dense_tile[2 * BLOCK_SIZE_N * (STRIDE_N + PADDING_HALF)];

    const int coostyleblkrowidx = d_blkcoostylerowidx[blk_idx];
    const int signbit = (coostyleblkrowidx >> 31) & 0x1;
    const int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;
    const size_t rowblkjstart = __ldg(&d_blkcoostylerowidx_colstart[blk_idx]);
    const int num_tiles = __ldg(&d_blkcoostylerowidx_colstart[blk_idx + 1]) - rowblkjstart;

    if (num_tiles == 0) return;

    const int4* ptr_A = (const int4*)d_Blockdense_Val + rowblkjstart * 32;
    const int* ptr_col_idx = d_tile_colidx + rowblkjstart * BLOCK_SIZE_N;
    
    const int rowid = thread_id >> 3;
    const int colid = (thread_id & 7) << 3;

    const int smem_offset_0 = rowid * (STRIDE_N + PADDING_HALF) + colid;
    const int smem_offset_1 = smem_offset_0 + 8 * (STRIDE_N + PADDING_HALF);
    const long long col_offset = dimN_index + colid;

    unsigned reg_b[4][2];
    float reg_c[4][4] = {0.0f};

    int prefetch_indices[2];
    MAT_VAL_TYPE* buffer_ptr = s_dense_tile;

    int idx_0 = __ldg(ptr_col_idx + rowid); 
    int idx_1 = __ldg(ptr_col_idx + rowid + 8); 

    long long glob_b_offset_0 = (long long)idx_0 * colB + col_offset;
    cp_async_int4(&buffer_ptr[smem_offset_0], &d_b[glob_b_offset_0]);

    long long glob_b_offset_1 = (long long)idx_1 * colB + col_offset;
    cp_async_int4(&buffer_ptr[smem_offset_1], &d_b[glob_b_offset_1]);

    if (num_tiles > 1) {
        prefetch_indices[0] = __ldg(ptr_col_idx + BLOCK_SIZE_N + rowid); 
        prefetch_indices[1] = __ldg(ptr_col_idx + BLOCK_SIZE_N + rowid + 8);
    }
    
    cp_async_commit();

    for (int i = 0; i < num_tiles; i++)
    {
        if (i < num_tiles - 1)
        {
            MAT_VAL_TYPE* smem_write_ptr = s_dense_tile + ((i + 1) & 1) * SMEM_TILE_STEP;
            
            long long next_glob_b_0 = (long long)prefetch_indices[0] * colB + col_offset;
            cp_async_int4(&smem_write_ptr[smem_offset_0], &d_b[next_glob_b_0]);

            long long next_glob_b_1 = (long long)prefetch_indices[1] * colB + col_offset;
            cp_async_int4(&smem_write_ptr[smem_offset_1], &d_b[next_glob_b_1]);
            
            cp_async_commit();

            if (i < num_tiles - 2) {
                int offset = (i + 2) * BLOCK_SIZE_N; 
                prefetch_indices[0] = __ldg(ptr_col_idx + offset + rowid);
                prefetch_indices[1] = __ldg(ptr_col_idx + offset + rowid + 8);
            }
        }

        if (i < num_tiles - 1) cp_async_wait_group<1>();
        else cp_async_wait_group<0>();
        
        __syncthreads();

        {
            MAT_VAL_TYPE* tile_ptr = s_dense_tile + (i & 1) * SMEM_TILE_STEP;

            int4 loaded_val = ptr_A[lane_id]; 
            unsigned *reg_a_ptr = (unsigned *)(&loaded_val);
            int tile_start_lane = (thread_id >> 5) * Tile_N; 
            int base_smem_idx = (lane_id & 15) * (STRIDE_N + PADDING_HALF) + tile_start_lane;

            ldmatrix_m16n8_x2_trans(reg_b[0], &tile_ptr[base_smem_idx + 0]);
            ldmatrix_m16n8_x2_trans(reg_b[1], &tile_ptr[base_smem_idx + 8]);
            ldmatrix_m16n8_x2_trans(reg_b[2], &tile_ptr[base_smem_idx + 16]);
            ldmatrix_m16n8_x2_trans(reg_b[3], &tile_ptr[base_smem_idx + 24]);

            mma_m16n8k16_f32(reg_c[0], reg_a_ptr, reg_b[0], reg_c[0]);
            mma_m16n8k16_f32(reg_c[1], reg_a_ptr, reg_b[1], reg_c[1]);
            mma_m16n8k16_f32(reg_c[2], reg_a_ptr, reg_b[2], reg_c[2]);
            mma_m16n8k16_f32(reg_c[3], reg_a_ptr, reg_b[3], reg_c[3]);
        }
        
        ptr_A += 32; 

        __syncthreads();
    }

    int group_id = lane_id >> 2;   // 0-7
    int tile_start_write = (thread_id >> 5) * Tile_N;
    
    long long row0_addr = (long long)(blki + group_id) * colB + dimN_index;
    long long row1_addr = row0_addr + 8 * colB;
    
    int col_base_start = tile_start_write + (lane_id & 3) * 2;

    if (signbit) 
    {
        #pragma unroll
        for (int k = 0; k < 4; k++) 
        {
            int current_col = col_base_start + (k << 3); 
            
            atomicAdd(&d_c[row0_addr + current_col], reg_c[k][0]);
            atomicAdd(&d_c[row0_addr + current_col + 1], reg_c[k][1]);
            atomicAdd(&d_c[row1_addr + current_col], reg_c[k][2]);
            atomicAdd(&d_c[row1_addr + current_col + 1], reg_c[k][3]);
        }
    } 
    else
    {
        #pragma unroll
        for (int k = 0; k < 4; k++) 
        {
            int current_col = col_base_start + (k << 3);
            
            float2* val2 = reinterpret_cast<float2*>(&reg_c[k][0]);
            *(float2*)(&d_c[row0_addr + current_col]) = val2[0];
            *(float2*)(&d_c[row1_addr + current_col]) = val2[1];
        }
    }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N>
__global__ void stir_spmm_cuda_kernel_v16_32(const int colB,
                                            const MAT_VAL_TYPE* __restrict__ d_Blockdense_Val,
                                            const int rowblkblock,
                                            const unsigned int* __restrict__ d_blkcoostylerowidx,
                                            const int* __restrict__ d_blkcoostylerowidx_colstart,
                                            const MAT_VAL_TYPE* __restrict__ d_b,
                                            MAT_ANS_TYPE* __restrict__ d_c,
                                            const int* __restrict__ d_tile_colidx)
{
    const int warp_id = threadIdx.x >> 5;  // 当前线程所属的 Warp (0, 1, 2, 3)
    const int lane_id = threadIdx.x & 31;  // Warp 内部的线程 ID (0~31)
    
    // const int blk_idx = blockIdx.x * 4 + warp_id;
    const int blk_idx = (blockIdx.z * gridDim.y + blockIdx.y) * 4 + warp_id;
    
    if (blk_idx >= rowblkblock) return;

    const int stride_32 = 32 + PADDING_HALF; // 40
    const int SMEM_TILE_STEP = BLOCK_SIZE_N * stride_32; 
    __shared__ MAT_VAL_TYPE s_dense_tile[4][2 * SMEM_TILE_STEP]; 

    MAT_VAL_TYPE* warp_smem = s_dense_tile[warp_id];

    const int coostyleblkrowidx = d_blkcoostylerowidx[blk_idx];
    const int signbit = (coostyleblkrowidx >> 31) & 0x1;
    const int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;
    const size_t rowblkjstart = __ldg(&d_blkcoostylerowidx_colstart[blk_idx]);
    const int num_tiles = __ldg(&d_blkcoostylerowidx_colstart[blk_idx + 1]) - rowblkjstart;

    if (num_tiles == 0) return;

    const int4* ptr_A = (const int4*)d_Blockdense_Val + rowblkjstart * 32;
    const int* ptr_col_idx = d_tile_colidx + rowblkjstart * BLOCK_SIZE_N;

    const int rowid_0 = lane_id >> 2;          // 0~7
    const int colid_0 = (lane_id & 3) << 3;    // 0, 8, 16, 24
    const int rowid_1 = rowid_0 + 8;           // 8~15
    const int colid_1 = colid_0;

    const int smem_offset_0 = rowid_0 * stride_32 + colid_0;
    const int smem_offset_1 = rowid_1 * stride_32 + colid_1;

    int idx_0 = __ldg(ptr_col_idx + rowid_0); 
    int idx_1 = __ldg(ptr_col_idx + rowid_1); 

    long long glob_b_offset_0 = (long long)idx_0 * 32 + colid_0;
    long long glob_b_offset_1 = (long long)idx_1 * 32 + colid_1;

    cp_async_int4(&warp_smem[smem_offset_0], &d_b[glob_b_offset_0]);
    cp_async_int4(&warp_smem[smem_offset_1], &d_b[glob_b_offset_1]);

    int prefetch_indices[2];
    if (num_tiles > 1) {
        prefetch_indices[0] = __ldg(ptr_col_idx + BLOCK_SIZE_N + rowid_0); 
        prefetch_indices[1] = __ldg(ptr_col_idx + BLOCK_SIZE_N + rowid_1);
    }
    
    cp_async_commit();

    unsigned reg_b[4][2];
    float reg_c[4][4] = {0.0f};

    for (int i = 0; i < num_tiles; i++)
    {
        if (i < num_tiles - 1)
        {
            MAT_VAL_TYPE* smem_write_ptr = warp_smem + ((i + 1) & 1) * SMEM_TILE_STEP;
            
            long long next_glob_b_0 = (long long)prefetch_indices[0] * 32 + colid_0;
            long long next_glob_b_1 = (long long)prefetch_indices[1] * 32 + colid_1;
            
            cp_async_int4(&smem_write_ptr[smem_offset_0], &d_b[next_glob_b_0]);
            cp_async_int4(&smem_write_ptr[smem_offset_1], &d_b[next_glob_b_1]);
            
            cp_async_commit();

            if (i < num_tiles - 2) {
                int offset = (i + 2) * BLOCK_SIZE_N; 
                prefetch_indices[0] = __ldg(ptr_col_idx + offset + rowid_0);
                prefetch_indices[1] = __ldg(ptr_col_idx + offset + rowid_1);
            }
        }

        if (i < num_tiles - 1) cp_async_wait_group<1>();
        else cp_async_wait_group<0>();
        
        __syncwarp(); 

        {
            MAT_VAL_TYPE* tile_ptr = warp_smem + (i & 1) * SMEM_TILE_STEP;

            int4 loaded_val = ptr_A[lane_id]; 
            unsigned *reg_a_ptr = (unsigned *)(&loaded_val);
            int base_smem_idx = (lane_id & 15) * stride_32;

            ldmatrix_m16n8_x2_trans(reg_b[0], &tile_ptr[base_smem_idx + 0]);
            ldmatrix_m16n8_x2_trans(reg_b[1], &tile_ptr[base_smem_idx + 8]);
            ldmatrix_m16n8_x2_trans(reg_b[2], &tile_ptr[base_smem_idx + 16]);
            ldmatrix_m16n8_x2_trans(reg_b[3], &tile_ptr[base_smem_idx + 24]);

            mma_m16n8k16_f32(reg_c[0], reg_a_ptr, reg_b[0], reg_c[0]);
            mma_m16n8k16_f32(reg_c[1], reg_a_ptr, reg_b[1], reg_c[1]);
            mma_m16n8k16_f32(reg_c[2], reg_a_ptr, reg_b[2], reg_c[2]);
            mma_m16n8k16_f32(reg_c[3], reg_a_ptr, reg_b[3], reg_c[3]);
        }
        
        ptr_A += 32; 
        __syncwarp(); 
    }

    int r0 = lane_id >> 2; 
    int c0 = (lane_id & 3) * 2; 

    long long row0_addr = (long long)(blki + r0) * 32;
    long long row1_addr = row0_addr + 8 * 32;
    
    if (signbit) 
    {
        #pragma unroll
        for (int k = 0; k < 4; k++) 
        {
            int current_col = k * 8 + c0; 
            atomicAdd(&d_c[row0_addr + current_col], reg_c[k][0]);
            atomicAdd(&d_c[row0_addr + current_col + 1], reg_c[k][1]);
            atomicAdd(&d_c[row1_addr + current_col], reg_c[k][2]);
            atomicAdd(&d_c[row1_addr + current_col + 1], reg_c[k][3]);
        }
    } 
    else
    {
        #pragma unroll
        for (int k = 0; k < 4; k++) 
        {
            int current_col = k * 8 + c0; 
            float2* val2 = reinterpret_cast<float2*>(&reg_c[k][0]);
            *(float2*)(&d_c[row0_addr + current_col]) = val2[0];
            *(float2*)(&d_c[row1_addr + current_col]) = val2[1];
        }
    }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N>
__global__ void stir_spmm_cuda_kernel_v64_tf32(int colB, 
                                          MAT_VAL_TYPE *d_Blockdense_Val,
                                          int rowblkblock,
                                          unsigned int *d_blkcoostylerowidx,
                                          int *d_blkcoostylerowidx_colstart,
                                          MAT_VAL_TYPE *d_b,
                                          MAT_ANS_TYPE *d_c,
                                          int *d_tile_colidx) 
{
    int group_id = threadIdx.x >> 6;
    int thread_id = threadIdx.x & 63;
    const int lane_id = thread_id & 31; 
    int blk_idx = blockIdx.z * gridDim.y + blockIdx.y;
    int dimN_index = blockIdx.x * STRIDE_N;

    if (blk_idx >= rowblkblock) return;

    int coostyleblkrowidx = d_blkcoostylerowidx[blk_idx];
    int signbit = (coostyleblkrowidx >> 31) & 0x1;
    int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;

    size_t rowblkjstart = __ldg(&d_blkcoostylerowidx_colstart[blk_idx]);
    int num_tiles = __ldg(&d_blkcoostylerowidx_colstart[blk_idx + 1]) - rowblkjstart;

    if (num_tiles == 0) return;

    int rowid = threadIdx.x >> 4; 
    int colid = (threadIdx.x & 15) << 2;    

    int smem_offset = rowid * stride_padded_float + colid;

    long long tile_start_offset = (long long)rowblkjstart * 1024; 
    const float4* ptr_A = (const float4*)(&d_Blockdense_Val[tile_start_offset + group_id * 256]);

    const int SMEM_TILE_STEP = BLOCK_SIZE_N * stride_padded_float;
    __shared__ float s_dense_tile[2 * SMEM_TILE_STEP];

    int original_row = __ldg(&d_tile_colidx[rowblkjstart * BLOCK_SIZE_N + rowid]);
    int prefetch_index = 0;

    long long global_offset = (long long)original_row * colB + dimN_index + colid;
    cp_async_int4(&s_dense_tile[smem_offset], &d_b[global_offset]);

    if (num_tiles > 1) {
        prefetch_index = __ldg(&d_tile_colidx[(rowblkjstart + 1) * BLOCK_SIZE_N + rowid]);
    }
    cp_async_commit();

    float reg_c[4][4] = {0.0f}; 

    for (int i = 0; i < num_tiles; i++)
    {
        if (i < num_tiles - 1)
        {
            float* smem_write_ptr = s_dense_tile + ((i + 1) & 1) * SMEM_TILE_STEP;
            long long global_offset_next = (long long)prefetch_index * colB + dimN_index + colid;
            
            cp_async_int4(&smem_write_ptr[smem_offset], &d_b[global_offset_next]);
            cp_async_commit();

            if (i < num_tiles - 2) {
                prefetch_index = __ldg(&d_tile_colidx[(rowblkjstart + i + 2) * BLOCK_SIZE_N + rowid]);
            }
        }

        if (i < num_tiles - 1) cp_async_wait_group<1>();
        else cp_async_wait_group<0>();
        
        __syncthreads();

        {
            float* tile_ptr = s_dense_tile + (i & 1) * SMEM_TILE_STEP;
            unsigned *smem_u = (unsigned*)tile_ptr;

            unsigned int reg_b[4][2]; 
            
            int tile_start_lane = (thread_id >> 5) * Tile_N; 
            int local_row = lane_id & 3; 
            int local_col = lane_id >> 2; 

            int base_idx0 = local_row * stride_padded_float + local_col + tile_start_lane;
            int base_idx1 = base_idx0 + 4 * stride_padded_float;
            
            reg_b[0][0] = smem_u[base_idx0];
            reg_b[0][1] = smem_u[base_idx1];
            reg_b[1][0] = smem_u[base_idx0 + 8];
            reg_b[1][1] = smem_u[base_idx1 + 8];
            reg_b[2][0] = smem_u[base_idx0 + 16];
            reg_b[2][1] = smem_u[base_idx1 + 16];
            reg_b[3][0] = smem_u[base_idx0 + 24];
            reg_b[3][1] = smem_u[base_idx1 + 24];
            
            float4 loaded_val_0 = *(ptr_A + lane_id); 
            unsigned *reg_a_ptr_0 = (unsigned *)(&loaded_val_0);
            
            mma_m16n8k8_tf32(reg_c[0], reg_a_ptr_0, reg_b[0], reg_c[0]);
            mma_m16n8k8_tf32(reg_c[1], reg_a_ptr_0, reg_b[1], reg_c[1]);
            mma_m16n8k8_tf32(reg_c[2], reg_a_ptr_0, reg_b[2], reg_c[2]);
            mma_m16n8k8_tf32(reg_c[3], reg_a_ptr_0, reg_b[3], reg_c[3]);

            base_idx0 = base_idx0 + 8 * stride_padded_float;
            base_idx1 = base_idx1 + 8 * stride_padded_float;

            reg_b[0][0] = smem_u[base_idx0];
            reg_b[0][1] = smem_u[base_idx1];
            reg_b[1][0] = smem_u[base_idx0 + 8];
            reg_b[1][1] = smem_u[base_idx1 + 8];
            reg_b[2][0] = smem_u[base_idx0 + 16];
            reg_b[2][1] = smem_u[base_idx1 + 16];
            reg_b[3][0] = smem_u[base_idx0 + 24];
            reg_b[3][1] = smem_u[base_idx1 + 24];
            
            float4 loaded_val_1 = *(ptr_A + 32 + lane_id); 
            unsigned *reg_a_ptr_1 = (unsigned *)(&loaded_val_1);

            mma_m16n8k8_tf32(reg_c[0], reg_a_ptr_1, reg_b[0], reg_c[0]);
            mma_m16n8k8_tf32(reg_c[1], reg_a_ptr_1, reg_b[1], reg_c[1]);
            mma_m16n8k8_tf32(reg_c[2], reg_a_ptr_1, reg_b[2], reg_c[2]);
            mma_m16n8k8_tf32(reg_c[3], reg_a_ptr_1, reg_b[3], reg_c[3]);
        }
        ptr_A += 256; 

        __syncthreads();
    }

    size_t global_row = blki + group_id * 16;
    int warp_col_start = (thread_id >> 5) * Tile_N;        
    int c0 = (lane_id & 3) * 2;

    long long row0_addr = (long long)(global_row + (lane_id >> 2)) * colB + dimN_index;
    long long row1_addr = (long long)(global_row + (lane_id >> 2) + 8) * colB + dimN_index;

    if (signbit) 
    {
        #pragma unroll
        for (int k = 0; k < 4; k++) 
        {
            int col_base = warp_col_start + k * 8;
            atomicAdd(&d_c[row0_addr + col_base + c0], reg_c[k][0]);
            atomicAdd(&d_c[row0_addr + col_base + c0 + 1], reg_c[k][1]);
            atomicAdd(&d_c[row1_addr + col_base + c0], reg_c[k][2]);
            atomicAdd(&d_c[row1_addr + col_base + c0 + 1], reg_c[k][3]);
        }
    } 
    else
    {
        #pragma unroll
        for (int k = 0; k < 4; k++) 
        {
            int col_base = warp_col_start + k * 8;
            float2* val2 = reinterpret_cast<float2*>(&reg_c[k][0]);
            *(float2*)(&d_c[row0_addr + col_base + c0]) = val2[0];
            *(float2*)(&d_c[row1_addr + col_base + c0]) = val2[1];
        }
    }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N>
__global__ void stir_spmm_cuda_kernel_v64_tf32_32(int colB,
                                          MAT_VAL_TYPE *d_Blockdense_Val,
                                          int rowblkblock,
                                          unsigned int *d_blkcoostylerowidx,
                                          int *d_blkcoostylerowidx_colstart,
                                          MAT_VAL_TYPE *d_b,
                                          MAT_ANS_TYPE *d_c,
                                          int *d_tile_colidx) 
{
    int group_id = threadIdx.x >> 5;      
    const int lane_id = threadIdx.x & 31; 

    // int blk_idx = blockIdx.x;
    int blk_idx = blockIdx.z * gridDim.y + blockIdx.y;

    if (blk_idx >= rowblkblock) return;

    int coostyleblkrowidx = d_blkcoostylerowidx[blk_idx];
    int signbit = (coostyleblkrowidx >> 31) & 0x1;
    int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;

    size_t rowblkjstart = __ldg(&d_blkcoostylerowidx_colstart[blk_idx]);
    int num_tiles = __ldg(&d_blkcoostylerowidx_colstart[blk_idx + 1]) - rowblkjstart;

    if (num_tiles == 0) return;

    const int stride_32 = Tile_N + PADDING_FLOAT; 
    const int SMEM_TILE_STEP = BLOCK_SIZE_N * stride_32; 
    __shared__ float s_dense_tile[2 * SMEM_TILE_STEP];

    int rowid = threadIdx.x >> 3;    
    int colid = (threadIdx.x & 7) << 2; 

    int smem_offset = rowid * stride_32 + colid;
    long long tile_start_offset = (long long)rowblkjstart * 1024; 
    const float4* ptr_A = (const float4*)(&d_Blockdense_Val[tile_start_offset + group_id * 256]);

    int original_row = __ldg(&d_tile_colidx[rowblkjstart * BLOCK_SIZE_N + rowid]);
    int prefetch_index = 0;

    long long global_offset = (long long)original_row * 32 + colid;
    cp_async_int4(&s_dense_tile[smem_offset], &d_b[global_offset]);

    if (num_tiles > 1) {
        prefetch_index = __ldg(&d_tile_colidx[(rowblkjstart + 1) * BLOCK_SIZE_N + rowid]);
    }
    cp_async_commit();

    float reg_c[4][4] = {0.0f}; 

    #pragma unroll
    for (int i = 0; i < num_tiles; i++)
    {
        if (i < num_tiles - 1)
        {
            float* smem_write_ptr = s_dense_tile + ((i + 1) & 1) * SMEM_TILE_STEP;
            long long global_offset_next = (long long)prefetch_index * 32 + colid;
            
            cp_async_int4(&smem_write_ptr[smem_offset], &d_b[global_offset_next]);
            cp_async_commit();

            if (i < num_tiles - 2) {
                prefetch_index = __ldg(&d_tile_colidx[(rowblkjstart + i + 2) * BLOCK_SIZE_N + rowid]);
            }
        }

        if (i < num_tiles - 1) cp_async_wait_group<1>();
        else cp_async_wait_group<0>();
        
        __syncthreads();

        {
            float* tile_ptr = s_dense_tile + (i & 1) * SMEM_TILE_STEP;
            unsigned *smem_u = (unsigned*)tile_ptr;

            unsigned int reg_b[4][2]; 
            
            int local_row = lane_id & 3; 
            int local_col = lane_id >> 2; 

            int base_idx0 = local_row * stride_32 + local_col;
            int base_idx1 = base_idx0 + 4 * stride_32;
            
            reg_b[0][0] = smem_u[base_idx0];
            reg_b[0][1] = smem_u[base_idx1];
            reg_b[1][0] = smem_u[base_idx0 + 8];
            reg_b[1][1] = smem_u[base_idx1 + 8];
            reg_b[2][0] = smem_u[base_idx0 + 16];
            reg_b[2][1] = smem_u[base_idx1 + 16];
            reg_b[3][0] = smem_u[base_idx0 + 24];
            reg_b[3][1] = smem_u[base_idx1 + 24];
            
            float4 loaded_val_0 = *(ptr_A + lane_id); 
            unsigned *reg_a_ptr_0 = (unsigned *)(&loaded_val_0);
            
            mma_m16n8k8_tf32(reg_c[0], reg_a_ptr_0, reg_b[0], reg_c[0]);
            mma_m16n8k8_tf32(reg_c[1], reg_a_ptr_0, reg_b[1], reg_c[1]);
            mma_m16n8k8_tf32(reg_c[2], reg_a_ptr_0, reg_b[2], reg_c[2]);
            mma_m16n8k8_tf32(reg_c[3], reg_a_ptr_0, reg_b[3], reg_c[3]);

            base_idx0 = base_idx0 + 8 * stride_32;
            base_idx1 = base_idx1 + 8 * stride_32;

            reg_b[0][0] = smem_u[base_idx0];
            reg_b[0][1] = smem_u[base_idx1];
            reg_b[1][0] = smem_u[base_idx0 + 8];
            reg_b[1][1] = smem_u[base_idx1 + 8];
            reg_b[2][0] = smem_u[base_idx0 + 16];
            reg_b[2][1] = smem_u[base_idx1 + 16];
            reg_b[3][0] = smem_u[base_idx0 + 24];
            reg_b[3][1] = smem_u[base_idx1 + 24];
            
            float4 loaded_val_1 = *(ptr_A + 32 + lane_id); 
            unsigned *reg_a_ptr_1 = (unsigned *)(&loaded_val_1);

            mma_m16n8k8_tf32(reg_c[0], reg_a_ptr_1, reg_b[0], reg_c[0]);
            mma_m16n8k8_tf32(reg_c[1], reg_a_ptr_1, reg_b[1], reg_c[1]);
            mma_m16n8k8_tf32(reg_c[2], reg_a_ptr_1, reg_b[2], reg_c[2]);
            mma_m16n8k8_tf32(reg_c[3], reg_a_ptr_1, reg_b[3], reg_c[3]);
        }
        ptr_A += 256; 

        __syncthreads();
    }

    size_t global_row = blki + group_id * 16;
    int c0 = (lane_id & 3) * 2;

    long long row0_addr = (long long)(global_row + (lane_id >> 2)) * 32;
    long long row1_addr = (long long)(global_row + (lane_id >> 2) + 8) * 32;

    if (signbit) 
    {
        #pragma unroll
        for (int k = 0; k < 4; k++) 
        {
            int col_base = k * 8;
            atomicAdd(&d_c[row0_addr + col_base + c0], reg_c[k][0]);
            atomicAdd(&d_c[row0_addr + col_base + c0 + 1], reg_c[k][1]);
            atomicAdd(&d_c[row1_addr + col_base + c0], reg_c[k][2]);
            atomicAdd(&d_c[row1_addr + col_base + c0 + 1], reg_c[k][3]);
        }
    } 
    else
    {
        #pragma unroll
        for (int k = 0; k < 4; k++) 
        {
            int col_base = k * 8;
            float2* val2 = reinterpret_cast<float2*>(&reg_c[k][0]);
            *(float2*)(&d_c[row0_addr + col_base + c0]) = val2[0];
            *(float2*)(&d_c[row1_addr + col_base + c0]) = val2[1];
        }
    }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N>
__global__ void stir_spmm_cuda_kernel_v64(int colB, 
                                          MAT_VAL_TYPE *d_Blockdense_Val,
                                          int rowblkblock,
                                          unsigned int *d_blkcoostylerowidx,
                                          int *d_blkcoostylerowidx_colstart,
                                          MAT_VAL_TYPE *d_b,
                                          MAT_ANS_TYPE *d_c,
                                          int *d_tile_colidx) 
{
    int group_id = threadIdx.x >> 6;
    int thread_id = threadIdx.x & 63;
    const int lane_id = thread_id % 32;
    int blk_idx = blockIdx.z * gridDim.y + blockIdx.y;
    int dimN_index = blockIdx.x * STRIDE_N;

    if (blk_idx >= rowblkblock) return;

    int coostyleblkrowidx = d_blkcoostylerowidx[blk_idx];
    int signbit = (coostyleblkrowidx >> 31) & 0x1;
    int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;

    size_t rowblkjstart = __ldg(&d_blkcoostylerowidx_colstart[blk_idx]);
    int num_tiles = __ldg(&d_blkcoostylerowidx_colstart[blk_idx + 1]) - rowblkjstart;

    if (num_tiles == 0) return;

    int b_row_id = threadIdx.x >> 3;      
    int b_col_id = (threadIdx.x & 7) << 3; 
    int smem_offset = b_row_id * stride_padded_half + b_col_id;
    
    int row_idx_b = lane_id % 16;
    int tile_start_B = (thread_id >> 5) * Tile_N;
    int base_smem_idx = row_idx_b * stride_padded_half + tile_start_B;

    long long tile_start_offset = (long long)rowblkjstart * 2048; 
    const int4* ptr_A = (const int4*)(&d_Blockdense_Val[tile_start_offset + group_id * 512]);

    const int SMEM_TILE_STEP = BLOCK_SIZE_N * stride_padded_half;
    __shared__ MAT_VAL_TYPE s_dense_tile[2 * SMEM_TILE_STEP];
    int original_row = __ldg(&d_tile_colidx[rowblkjstart * BLOCK_SIZE_N + b_row_id]);

    MAT_VAL_TYPE* smem_ptr_0 = s_dense_tile; 
    long long global_offset = (long long)original_row * colB + dimN_index + b_col_id;
    cp_async_int4(&smem_ptr_0[smem_offset], &d_b[global_offset]);
    cp_async_commit();

    int next_original_row = 0;
    if (num_tiles > 1) {
        next_original_row = __ldg(&d_tile_colidx[(rowblkjstart + 1) * BLOCK_SIZE_N + b_row_id]);
    }

    float reg_c[4][4] = {}; 

    #pragma unroll
    for (int i = 0; i < num_tiles; i++)
    {
        if (i < num_tiles - 1)
        {
            int write_buffer_idx = (i + 1) % 2;
            MAT_VAL_TYPE* smem_write_ptr = s_dense_tile + write_buffer_idx * SMEM_TILE_STEP;
            long long global_offset_next = (long long)next_original_row * colB + dimN_index + b_col_id;
            
            cp_async_int4(&smem_write_ptr[smem_offset], &d_b[global_offset_next]);
            cp_async_commit();

            if (i < num_tiles - 2) {
                int tile_ptr_next = rowblkjstart + i + 2;
                next_original_row = __ldg(&d_tile_colidx[tile_ptr_next * BLOCK_SIZE_N + b_row_id]);
            }
        }

        if (i < num_tiles - 1) cp_async_wait_group<1>();
        else cp_async_wait_group<0>();
        
        __syncthreads();

        {
            unsigned reg_b[4][2];
            int read_buffer_idx = i % 2;
            MAT_VAL_TYPE* buffer_ptr = s_dense_tile + read_buffer_idx * SMEM_TILE_STEP;
            
            int4 loaded_val_0 = *(ptr_A + lane_id); 
            int4 loaded_val_1 = *(ptr_A + 32 + lane_id); 
            
            unsigned *reg_a_ptr_0 = (unsigned *)(&loaded_val_0);
            unsigned *reg_a_ptr_1 = (unsigned *)(&loaded_val_1);
            
            ldmatrix_m16n8_x2_trans(reg_b[0], &buffer_ptr[base_smem_idx + 0]);
            ldmatrix_m16n8_x2_trans(reg_b[1], &buffer_ptr[base_smem_idx + 8]);
            ldmatrix_m16n8_x2_trans(reg_b[2], &buffer_ptr[base_smem_idx + 16]);
            ldmatrix_m16n8_x2_trans(reg_b[3], &buffer_ptr[base_smem_idx + 24]);

            mma_m16n8k16_f32(reg_c[0], reg_a_ptr_0, reg_b[0], reg_c[0]);
            mma_m16n8k16_f32(reg_c[1], reg_a_ptr_0, reg_b[1], reg_c[1]);
            mma_m16n8k16_f32(reg_c[2], reg_a_ptr_0, reg_b[2], reg_c[2]);
            mma_m16n8k16_f32(reg_c[3], reg_a_ptr_0, reg_b[3], reg_c[3]);

            int base_smem_idx_2 = base_smem_idx + 16 * stride_padded_half;
            ldmatrix_m16n8_x2_trans(reg_b[0], &buffer_ptr[base_smem_idx_2 + 0]);
            ldmatrix_m16n8_x2_trans(reg_b[1], &buffer_ptr[base_smem_idx_2 + 8]);
            ldmatrix_m16n8_x2_trans(reg_b[2], &buffer_ptr[base_smem_idx_2 + 16]);
            ldmatrix_m16n8_x2_trans(reg_b[3], &buffer_ptr[base_smem_idx_2 + 24]);

            mma_m16n8k16_f32(reg_c[0], reg_a_ptr_1, reg_b[0], reg_c[0]);
            mma_m16n8k16_f32(reg_c[1], reg_a_ptr_1, reg_b[1], reg_c[1]);
            mma_m16n8k16_f32(reg_c[2], reg_a_ptr_1, reg_b[2], reg_c[2]);
            mma_m16n8k16_f32(reg_c[3], reg_a_ptr_1, reg_b[3], reg_c[3]);
        }
        
        ptr_A += 256;

        __syncthreads();
    }

    size_t global_row = blki + group_id * 16;
    int warp_col_start = (thread_id >> 5) * Tile_N;
    int r0 = lane_id >> 2;    
    int r1 = r0 + 8;         
    int c0 = (lane_id & 3) * 2;

    long long row0_addr = (long long)(global_row + r0) * colB + dimN_index;
    long long row1_addr = (long long)(global_row + r1) * colB + dimN_index;

    if (signbit) 
    {
        #pragma unroll
        for (int k = 0; k < 4; k++) 
        {
            int col_base = warp_col_start + k * 8;
            atomicAdd(&d_c[row0_addr + col_base + c0], reg_c[k][0]);
            atomicAdd(&d_c[row0_addr + col_base + c0 + 1], reg_c[k][1]);
            atomicAdd(&d_c[row1_addr + col_base + c0], reg_c[k][2]);
            atomicAdd(&d_c[row1_addr + col_base + c0 + 1], reg_c[k][3]);
        }
    } 
    else
    {
        #pragma unroll
        for (int k = 0; k < 4; k++) 
        {
            int col_base = warp_col_start + k * 8;
            float2* val2 = reinterpret_cast<float2*>(&reg_c[k][0]);
            *(float2*)(&d_c[row0_addr + col_base + c0]) = val2[0];
            *(float2*)(&d_c[row1_addr + col_base + c0]) = val2[1];
        }
    }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N>
__global__ void stir_spmm_cuda_kernel_v64_32(int colB, 
                                          MAT_VAL_TYPE *d_Blockdense_Val,
                                          int rowblkblock,
                                          unsigned int *d_blkcoostylerowidx,
                                          int *d_blkcoostylerowidx_colstart,
                                          MAT_VAL_TYPE *d_b,
                                          MAT_ANS_TYPE *d_c,
                                          int *d_tile_colidx) 
{    
    const int lane_id = threadIdx.x & 31; 
    
    // int blk_idx = blockIdx.x;
    int blk_idx = blockIdx.z * gridDim.y + blockIdx.y;

    if (blk_idx >= rowblkblock) return;

    int coostyleblkrowidx = d_blkcoostylerowidx[blk_idx];
    int signbit = (coostyleblkrowidx >> 31) & 0x1;
    int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;

    size_t rowblkjstart = __ldg(&d_blkcoostylerowidx_colstart[blk_idx]);
    int num_tiles = __ldg(&d_blkcoostylerowidx_colstart[blk_idx + 1]) - rowblkjstart;

    if (num_tiles == 0) return;

    const int stride = Tile_N + PADDING_HALF; // 32 + 8 = 40
    const int SMEM_TILE_STEP = BLOCK_SIZE_N * stride;
    __shared__ MAT_VAL_TYPE s_dense_tile[2 * SMEM_TILE_STEP];

    int b_row_id = threadIdx.x >> 2;     
    int b_col_id = (threadIdx.x & 3) << 3; 
    int smem_offset = b_row_id * stride + b_col_id;
    
    int row_idx_b = lane_id % 16;
    int base_smem_idx = row_idx_b * stride;

    long long tile_start_offset = (long long)rowblkjstart * 2048; 
    const int4* ptr_A = (const int4*)(&d_Blockdense_Val[tile_start_offset + (threadIdx.x >> 5) * 512]);

    int original_row = __ldg(&d_tile_colidx[rowblkjstart * BLOCK_SIZE_N + b_row_id]);
    MAT_VAL_TYPE* smem_ptr_0 = s_dense_tile; 
    
    long long global_offset = (long long)original_row * 32 + b_col_id;
    cp_async_int4(&smem_ptr_0[smem_offset], &d_b[global_offset]);
    cp_async_commit();

    int next_original_row = 0;
    if (num_tiles > 1) {
        next_original_row = __ldg(&d_tile_colidx[(rowblkjstart + 1) * BLOCK_SIZE_N + b_row_id]);
    }

    float reg_c[4][4] = {}; 

    #pragma unroll
    for (int i = 0; i < num_tiles; i++)
    {
        if (i < num_tiles - 1)
        {
            int write_buffer_idx = (i + 1) % 2;
            MAT_VAL_TYPE* smem_write_ptr = s_dense_tile + write_buffer_idx * SMEM_TILE_STEP;
            long long global_offset_next = (long long)next_original_row * 32 + b_col_id;
            
            cp_async_int4(&smem_write_ptr[smem_offset], &d_b[global_offset_next]);
            cp_async_commit();

            if (i < num_tiles - 2) {
                int tile_ptr_next = rowblkjstart + i + 2;
                next_original_row = __ldg(&d_tile_colidx[tile_ptr_next * BLOCK_SIZE_N + b_row_id]);
            }
        }

        if (i < num_tiles - 1) cp_async_wait_group<1>();
        else cp_async_wait_group<0>();
        
        __syncthreads();

        {
            unsigned reg_b[4][2];
            int read_buffer_idx = i % 2;
            MAT_VAL_TYPE* buffer_ptr = s_dense_tile + read_buffer_idx * SMEM_TILE_STEP;
            
            int4 loaded_val = *(ptr_A + lane_id); 
            unsigned *reg_a_ptr = (unsigned *)(&loaded_val);
            
            ldmatrix_m16n8_x2_trans(reg_b[0], &buffer_ptr[base_smem_idx + 0]);
            ldmatrix_m16n8_x2_trans(reg_b[1], &buffer_ptr[base_smem_idx + 8]);
            ldmatrix_m16n8_x2_trans(reg_b[2], &buffer_ptr[base_smem_idx + 16]);
            ldmatrix_m16n8_x2_trans(reg_b[3], &buffer_ptr[base_smem_idx + 24]);

            mma_m16n8k16_f32(reg_c[0], reg_a_ptr, reg_b[0], reg_c[0]);
            mma_m16n8k16_f32(reg_c[1], reg_a_ptr, reg_b[1], reg_c[1]);
            mma_m16n8k16_f32(reg_c[2], reg_a_ptr, reg_b[2], reg_c[2]);
            mma_m16n8k16_f32(reg_c[3], reg_a_ptr, reg_b[3], reg_c[3]);

            loaded_val = *(ptr_A + 32 + lane_id); 
            reg_a_ptr = (unsigned *)(&loaded_val);
            
            int base_smem_idx_2 = base_smem_idx + 16 * stride;
            ldmatrix_m16n8_x2_trans(reg_b[0], &buffer_ptr[base_smem_idx_2 + 0]);
            ldmatrix_m16n8_x2_trans(reg_b[1], &buffer_ptr[base_smem_idx_2 + 8]);
            ldmatrix_m16n8_x2_trans(reg_b[2], &buffer_ptr[base_smem_idx_2 + 16]);
            ldmatrix_m16n8_x2_trans(reg_b[3], &buffer_ptr[base_smem_idx_2 + 24]);

            mma_m16n8k16_f32(reg_c[0], reg_a_ptr, reg_b[0], reg_c[0]);
            mma_m16n8k16_f32(reg_c[1], reg_a_ptr, reg_b[1], reg_c[1]);
            mma_m16n8k16_f32(reg_c[2], reg_a_ptr, reg_b[2], reg_c[2]);
            mma_m16n8k16_f32(reg_c[3], reg_a_ptr, reg_b[3], reg_c[3]);
        }
        
        ptr_A += 256;

        __syncthreads();
    }

    size_t global_row = blki + (threadIdx.x >> 5) * 16;       
    int c0 = (lane_id & 3) * 2;
    long long row0_addr = (long long)(global_row + (lane_id >> 2)) * 32;
    long long row1_addr = row0_addr + 256;

    if (signbit) 
    {
        #pragma unroll
        for (int k = 0; k < 4; k++) 
        {
            int col_base = k * 8; 
            atomicAdd(&d_c[row0_addr + col_base + c0], reg_c[k][0]);
            atomicAdd(&d_c[row0_addr + col_base + c0 + 1], reg_c[k][1]);
            atomicAdd(&d_c[row1_addr + col_base + c0], reg_c[k][2]);
            atomicAdd(&d_c[row1_addr + col_base + c0 + 1], reg_c[k][3]);
        }
    } 
    else
    {
        #pragma unroll
        for (int k = 0; k < 4; k++) 
        {
            int col_base = k * 8;
            float2* val2 = reinterpret_cast<float2*>(&reg_c[k][0]);
            *(float2*)(&d_c[row0_addr + col_base + c0]) = val2[0];
            *(float2*)(&d_c[row1_addr + col_base + c0]) = val2[1];
        }
    }
}

void switch_tilespmm_cuda_v8(int rowblkblock, int colB, 
                            MAT_VAL_TYPE *d_Blockdense_Val, 
                            unsigned int *d_blkcoostylerowidx, 
                            int *d_blkcoostylerowidx_colstart,
                            MAT_VAL_TYPE *d_b, MAT_ANS_TYPE *d_c, 
                            int *d_tile_colidx, cudaStream_t stream = 0)
{
    if (colB == 32) 
    {
        int warps_per_block = 4;
        int blocknum = (rowblkblock + warps_per_block - 1) / warps_per_block;
        int grid_z = (blocknum + grid_y_limit - 1) / grid_y_limit;
        dim3 grid_dim(1, grid_y_limit, grid_z);
        dim3 block_dim(128, 1, 1);
#ifdef TFLOAT
        stir_spmm_cuda_kernel_v8_tf32_32<8,8><<<grid_dim, block_dim, 0, stream>>>(colB, d_Blockdense_Val, rowblkblock, d_blkcoostylerowidx, 
                                                                                d_blkcoostylerowidx_colstart, d_b, d_c, d_tile_colidx);
#else   
        stir_spmm_cuda_kernel_v8_32<8,16><<<grid_dim, block_dim, 0, stream>>>(colB, d_Blockdense_Val, rowblkblock, d_blkcoostylerowidx, 
                                                                            d_blkcoostylerowidx_colstart, d_b, d_c, d_tile_colidx);
#endif
    }
    else
    {
        int num_blocks_x = (colB + 63) / 64;
        int grid_z = (rowblkblock + grid_y_limit - 1) / grid_y_limit;
        dim3 grid_dim(num_blocks_x, grid_y_limit, grid_z);
        dim3 block_dim(64, 1, 1); 
#ifdef TFLOAT
        stir_spmm_cuda_kernel_v8_tf32<8,8><<<grid_dim, block_dim, 0, stream>>>(colB, d_Blockdense_Val, rowblkblock, d_blkcoostylerowidx, 
                                                                                d_blkcoostylerowidx_colstart, d_b, d_c, d_tile_colidx);
#else
        stir_spmm_cuda_kernel_v8<8,16><<<grid_dim, block_dim, 0, stream>>>(colB, d_Blockdense_Val, rowblkblock, d_blkcoostylerowidx, 
                                                                            d_blkcoostylerowidx_colstart, d_b, d_c, d_tile_colidx);   
#endif      
    }
}

void switch_tilespmm_cuda_v16(int rowblkblock, int colB, 
                            MAT_VAL_TYPE *d_Blockdense_Val, 
                            unsigned int *d_blkcoostylerowidx, 
                            int *d_blkcoostylerowidx_colstart,
                            MAT_VAL_TYPE *d_b, MAT_ANS_TYPE *d_c, 
                            int *d_tile_colidx, cudaStream_t stream = 0)
{
    if (colB == 32) 
    {
        int warps_per_block = 4;
        int blocknum = (rowblkblock + warps_per_block - 1) / warps_per_block;
        int grid_z = (blocknum + grid_y_limit - 1) / grid_y_limit;
        dim3 grid_dim(1, grid_y_limit, grid_z);
        dim3 block_dim(128, 1, 1);
#ifdef TFLOAT
        stir_spmm_cuda_kernel_v16_tf32_32<16,8><<<grid_dim, block_dim, 0, stream>>>(colB, d_Blockdense_Val, rowblkblock, d_blkcoostylerowidx, 
                                                                                d_blkcoostylerowidx_colstart, d_b, d_c, d_tile_colidx);
#else
        stir_spmm_cuda_kernel_v16_32<16,16><<<grid_dim, block_dim, 0, stream>>>(colB, d_Blockdense_Val, rowblkblock, d_blkcoostylerowidx, 
                                                                            d_blkcoostylerowidx_colstart, d_b, d_c, d_tile_colidx);
#endif
    }
    else
    {
        int num_blocks_x = (colB + 63) / 64;
        int grid_z = (rowblkblock + grid_y_limit - 1) / grid_y_limit;
        dim3 grid_dim(num_blocks_x, grid_y_limit, grid_z);
        dim3 block_dim(64, 1, 1); 
#ifdef TFLOAT
        stir_spmm_cuda_kernel_v16_tf32<16,8><<<grid_dim, block_dim, 0, stream>>>(colB, d_Blockdense_Val, rowblkblock, d_blkcoostylerowidx, 
                                                                                d_blkcoostylerowidx_colstart, d_b, d_c, d_tile_colidx);
#else
        stir_spmm_cuda_kernel_v16<16,16><<<grid_dim, block_dim, 0, stream>>>(colB, d_Blockdense_Val, rowblkblock, d_blkcoostylerowidx, 
                                                                            d_blkcoostylerowidx_colstart, d_b, d_c, d_tile_colidx);
#endif
    }
}

void switch_tilespmm_cuda_v64(int rowblkblock, int colB, 
                            MAT_VAL_TYPE *d_Blockdense_Val, 
                            unsigned int *d_blkcoostylerowidx, 
                            int *d_blkcoostylerowidx_colstart, 
                            MAT_VAL_TYPE *d_b, MAT_ANS_TYPE *d_c, 
                            int *d_tile_colidx, cudaStream_t stream = 0) 
{
    if(colB == 32)
    {     
        int grid_z = (rowblkblock + grid_y_limit - 1) / grid_y_limit;
        dim3 grid_dim(1, grid_y_limit, grid_z);
        dim3 block_dim(128, 1, 1);
#ifdef TFLOAT
        stir_spmm_cuda_kernel_v64_tf32_32<64,16><<<grid_dim, block_dim, 0, stream>>>(colB, d_Blockdense_Val, rowblkblock, d_blkcoostylerowidx, 
                                                                                    d_blkcoostylerowidx_colstart, d_b, d_c, d_tile_colidx);
#else
        stir_spmm_cuda_kernel_v64_32<64,32><<<grid_dim, block_dim, 0, stream>>>(colB, d_Blockdense_Val, rowblkblock, d_blkcoostylerowidx, 
                                                                                d_blkcoostylerowidx_colstart, d_b, d_c, d_tile_colidx);
#endif
    }
    else
    {
        int num_blocks_x = (colB + 63) / 64; 
        int grid_z = (rowblkblock + grid_y_limit - 1) / grid_y_limit;
        dim3 grid_dim(num_blocks_x, grid_y_limit, grid_z);
        dim3 block_dim(256, 1, 1); 
#ifdef TFLOAT    
        stir_spmm_cuda_kernel_v64_tf32<64,16><<<grid_dim, block_dim, 0, stream>>>(colB, d_Blockdense_Val, rowblkblock, d_blkcoostylerowidx, 
                                                                                d_blkcoostylerowidx_colstart, d_b, d_c, d_tile_colidx);
#else
        stir_spmm_cuda_kernel_v64<64,32><<<grid_dim, block_dim, 0, stream>>>(colB, d_Blockdense_Val, rowblkblock, d_blkcoostylerowidx, 
                                                                            d_blkcoostylerowidx_colstart, d_b, d_c, d_tile_colidx);
#endif
    }
}

#endif