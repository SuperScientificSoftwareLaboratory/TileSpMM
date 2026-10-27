# TileSpMM

## Paper

This repository contains the code for our ICPP '26 paper, which received the **Best Paper Award**:

Hongwei Zeng, Shuqin Feng, Haocheng Lian, Yuechen Lu, and Weifeng Liu. 2026. [TileSpMM: A Variable-Size Tiled Algorithm for Sparse Matrix-Matrix Multiplication on Tensor Cores](https://doi.org/10.1145/3832810.3832918). In *Proceedings of the 55th International Conference on Parallel Processing (ICPP '26)*. Association for Computing Machinery.

## Introduction

TileSpMM accelerates sparse matrix–dense matrix multiplication (SpMM) on NVIDIA Tensor Cores. To address irregular sparsity and excessive zero padding with fixed-size tiles, it adapts to local sparsity patterns using 64-, 16-, and 8-row tiles, balancing data reuse and padding overhead. A prediction model selects partition thresholds, while dedicated data layouts and kernels support the different tile sizes. Adaptive task partitioning and prefetching help balance workloads and hide memory latency. This implementation supports FP16 and TF32 inputs with FP32 accumulation and provides performance comparisons with cuSPARSE.

## Installation

### Requirements

- An NVIDIA GPU with compute capability 8.0 or later and a compatible driver.
- CUDA Toolkit with `nvcc`, cuSPARSE, and cuBLAS. The paper uses CUDA 12.8.

### Build

Run the following steps from the repository root.

**1. Configure the GPU architecture** in [`build_config.mk`](build_config.mk):

```make
CUDA_ARCH = 120
```

Use `120` for RTX 5090, `90` for H100, or `80` for A100. Set `CUDA_INSTALL_PATH` in the same file if needed, and ensure `nvcc` is available in `PATH`.

**2. Prepare the platform** on the target GPU:

```sh
sh prepare_prefetch.sh
```

The default GPU ID is 0. To use GPU 1, run `sh prepare_prefetch.sh 1`.

**3. Compile both precisions:**

```sh
make all
```

Alternatively, use `make fp16` or `make tf32` to build one precision. Executables are placed in `bin/`.

Repeat preparation and compilation after changing the GPU platform or architecture.

## Directory Structure

| Directory | Contents |
| --- | --- |
| `src/` | Core implementation, grouped by purpose below. |
| `pre/` | Kernel resource probes and build scripts for platform preparation. |
| `lib/` | Generated model library, device and kernel resource information, and probe executables. |
| `bin/` | Compiled FP16 and TF32 executables. |

The main files in `src/` are organized as follows:

- `main.cu`: Program entry, TileSpMM and cuSPARSE benchmarking, result checking, and CSV output.
- `hetero_split.h`, `csr2tile.h`: Variable-size tile partitioning, CSR-to-tile conversion, load balancing, and kernel dispatch.
- `tilespmm_cuda.h`, `utils.h`: FP16/TF32 CUDA kernels, GPU instruction helpers, and utility functions.
- `model.h`, `ml_predict.c`, `ml_predict.h`, `model_features.h`: Trained threshold prediction models, inference interfaces, and input feature construction.
- `mmio.h`, `mmio_highlevel.h`: Matrix Market file loading.
- `common.h`, `format.h`, `error_check.h`: Shared definitions, tile data structures, and error checks.

## Execution

Input matrices use Matrix Market format (`.mtx`).

### Single matrix

```sh
./bin/tilespmm_fp16 -d 0 /path/to/matrix.mtx 32
./bin/tilespmm_tf32 -d 0 /path/to/matrix.mtx 128
```

`-d 0` selects GPU 0. The last argument is `colB`, the number of columns of the dense matrix B. Use the GPU selected during platform preparation.

### Test example

Run [`test.sh`](test.sh) to test the matrices in `data/`:

```sh
bash test.sh
```

## Results

Results append to `TileSpMM-FP16.csv` and `TileSpMM-TF32.csv` in the current working directory. Each row contains the following 19 fields in order:

```csv
filename,row,col,nnz,colB,tilenum64,tilenum16,tilenum8,gamma,delta,best_th64,best_th16,best_th8,tilespmm_pre_time_ms,tilespmm_time_ms,tilespmm_tflops,cusparse_pre_time_ms,cusparse_time_ms,cusparse_tflops
```

`filename` is the matrix file path; `row`, `col`, and `nnz` are the row count, column count, and nonzero count of matrix A, and `colB` is the column count of matrix B. `tilenum64`, `tilenum16`, and `tilenum8` are the tile counts for the three row tile sizes; `gamma` and `delta` are partition thresholds. `best_th64`, `best_th16`, and `best_th8` are the load-balancing thresholds selected by runtime tuning for the corresponding tile sizes. The last six columns record preprocessing time, runtime, and throughput for TileSpMM, followed by the same three metrics for cuSPARSE. Times are in milliseconds and throughput is in TFLOPS.

## Reference

If you use TileSpMM in your research, please cite:

```bibtex
@inproceedings{zeng2026tilespmm,
  author    = {Hongwei Zeng and Shuqin Feng and Haocheng Lian and
               Yuechen Lu and Weifeng Liu},
  title     = {{TileSpMM}: A Variable-Size Tiled Algorithm for Sparse
               Matrix-Matrix Multiplication on Tensor Cores},
  booktitle = {Proceedings of the 55th International Conference on Parallel Processing},
  series    = {ICPP '26},
  year      = {2026},
  doi       = {10.1145/3832810.3832918},
  url       = {https://doi.org/10.1145/3832810.3832918}
}
```

## License

This project is licensed under the GNU Affero General Public License, version 3.0 (AGPL-3.0). See [LICENSE](LICENSE).

## Contact

Hongwei Zeng: [hongwei@student.cup.edu.cn](mailto:hongwei@student.cup.edu.cn)
