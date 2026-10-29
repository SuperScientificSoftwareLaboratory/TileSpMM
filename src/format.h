#include "common.h"

// Tile 格式数据与当前使用的调度信息
typedef struct
{
    int tilem;              // 行块数
    int tilen;              // 列块数
    int tilenum;            // 总 Tile 数
    MAT_PTR_TYPE *tile_ptr; // 行块指针
    int *tile_nnz;          // 每个 Tile 的非零元计数及前缀偏移
    
    int *tile_colidx; 
    MAT_VAL_TYPE *Blockdense_Val; 
    size_t dnssize;      
    
    int rowblkblock;            
    unsigned int *blkcoostylerowidx;   
    int *blkcoostylerowidx_colstart;   
    int *orig_tile_row_idx;
    
} Tile_matrix;

// 切分阈值及其代价，用于候选排序
typedef struct {
    int TH;
    float cost;
} CostCand;

// 一个切分阈值对应的主机调度数组
typedef struct
{
    int TH;
    int rowblkblock;
    unsigned int *rows;
    int *colstart;
} Prefetch_schedule;

// 缓存最多四份调度，避免重复构建；数组由调度流程释放
typedef struct
{
    Prefetch_schedule entries[4];
    size_t bytes;
} Prefetch_cache;

// 三种行块转换时共用的临时空间，转换结束后统一释放
typedef struct
{
    unsigned thread;
    int colA;
    char *flag;
    int *valid_cols;
    int *col_to_rank;
    int *tile_count;
    int tile_count_capacity;
} Tile_workspace;

// 释放 Tile 数据及当前调度数组，矩阵对象本身由调用者释放
void Tile_destroy(Tile_matrix *matrix)
{
    if (matrix == NULL) return;
    if (matrix->tile_ptr) free(matrix->tile_ptr);
    if (matrix->tile_nnz) free(matrix->tile_nnz);
    if (matrix->tile_colidx) free(matrix->tile_colidx);
    if (matrix->Blockdense_Val) free(matrix->Blockdense_Val);
    if (matrix->blkcoostylerowidx) free(matrix->blkcoostylerowidx);
    if (matrix->blkcoostylerowidx_colstart) free(matrix->blkcoostylerowidx_colstart);
}
