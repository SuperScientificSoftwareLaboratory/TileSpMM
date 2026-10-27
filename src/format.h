#include "common.h"

typedef struct
{
    int tilem;              // 行块数
    int tilen;              // 列块数
    int tilenum;            // 总 Tile 数
    MAT_PTR_TYPE *tile_ptr; // 行块指针 (Row Pointer)
    int *tile_nnz;          // 每个 Tile 的非零元计数 (用于构建 offset)
    
    int *tile_colidx; 
    MAT_VAL_TYPE *Blockdense_Val; 
    size_t dnssize;      
    
    int rowblkblock;            
    unsigned int *blkcoostylerowidx;   
    int *blkcoostylerowidx_colstart;   
    int *orig_tile_row_idx;
    
} Tile_matrix;

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