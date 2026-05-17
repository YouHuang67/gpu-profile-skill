# Bottleneck Patterns -> GPU Code Mapping

This reference maps NCU metric patterns to likely source-code causes across GPU frameworks. Use this to correlate hardware metrics with source code during analysis.

**Certainty caveat**: All GPU DSLs (Triton, Numba, TileLang, nvcc) compile through multiple optimization passes to PTX/SASS. The mapping from source-level patterns to hardware behavior is approximate. Always qualify with "likely" or "possibly".

---

## Memory Bound Patterns

### Pattern: High Memory SOL + Low L1 Hit Rate
**Metrics**: `gpu__compute_memory_throughput` > 60%, `l1tex__t_sector_hit_rate` < 50%
**Likely cause**: Poor spatial locality in memory access
**Code to check, by framework**:

| Framework | Pattern to look for |
|-----------|-------------------|
| Triton | `tl.load(ptr + offsets)` with large stride; `tl.make_block_ptr` with suboptimal `order` param |
| CUDA (Numba) | `cuda.blockIdx.x * blockDim.x + threadIdx.x` with non-contiguous index; column-major on row-major data |
| CUDA (Raw) | `threadIdx.x + blockDim.x * blockIdx.x` used as column index (not row index) |
| TileLang | `T.Parallel(M, N)` with wrong loop nesting; `T.copy` with mismatched access patterns |

**Investigation**: Compare actual bytes read (`dram__bytes_read.sum`) with theoretical minimum. Ratio > 2x means significant waste.

### Pattern: High DRAM Throughput Near Peak
**Metrics**: `dram__throughput` > 80%
**Likely cause**: Kernel is genuinely bandwidth-limited (this may be optimal)
**Code to check**:
- Verify arithmetic intensity — elementwise/reduction kernels being memory-bound is expected
- Check for redundant loads (loading same data multiple times across threads)
- For attention kernels: verify KV cache reuse through tiling

### Pattern: High Sectors/Request (Coalescing Waste)
**Metrics**: `global_load_sectors_per_request` > 4.0 (derived: load_sectors / load_requests)
**Scale**: 1.0 = perfectly coalesced, 32.0 = worst case
**Likely cause**: Threads within a warp access non-contiguous 32-byte sectors
**Code to check, by framework**:

| Framework | Pattern to look for |
|-----------|-------------------|
| Triton | `tl.load` with strided offsets; column-major access on row-major data; gather with scattered indices |
| CUDA (Numba) | Non-sequential thread indexing; `A[row * N + col]` where col = threadIdx.x but row varies per warp |
| CUDA (Raw) | Strided global memory access; scatter/gather from sparse indices |
| TileLang | `T.copy` between misaligned buffers; scatter store patterns |

**Investigation**: Check if `global_store_sectors_per_request` is also high — if only loads are bad, focus on load patterns.

### Pattern: Low Coalescing Efficiency
**Metrics**: `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld` < 50%
**Likely cause**: Threads within a warp access non-contiguous memory
**Code to check**: Strided access patterns, gather operations with irregular indices, sparse attention index lookup.

---

## Compute Bound Patterns

### Pattern: High Compute SOL + No Tensor Core Usage
**Metrics**: `sm__throughput` > 60%, `sm__pipe_tensor_cycles_active` < 5%
**Likely cause**: Matrix operations not using tensor cores
**Code to check, by framework**:

| Framework | Pattern to look for |
|-----------|-------------------|
| Triton | `tl.dot(a, b)` — verify both operands are fp16/bf16. FP32 uses FP32 pipe, not tensor cores |
| CUDA (Numba) | Manual loop-based matmul instead of `numba.cuda.cublas` or warp-level MMA intrinsics |
| CUDA (Raw) | FP32 `__hmma` instead of `__hmma_f16`; missing `mma.sync` instructions |
| TileLang | `T.gemm` with FP32 accumulators; verify input buffers are fp16 |

### Pattern: High Compute SOL + High Tensor Core Usage
**Metrics**: `sm__throughput` > 60%, `sm__pipe_tensor_cycles_active` > 30%
**Likely cause**: Compute genuinely saturated (may be near-optimal)
**Code to check**: Verify tile sizes are multiples of 16 for TC efficiency; check if problem size is large enough to saturate the GPU.

### Pattern: High FP32 Pipe + Expected FP16/BF16 Workload
**Metrics**: `sm__inst_executed_pipe_fp32` high, tensor pipe near 0
**Likely cause**: Data type mismatch — inputs or intermediates accidentally in FP32
**Code to check**: Missing dtype conversions, load reading FP32 data when FP16 was intended, intermediate accumulations promoting to FP32 unnecessarily.

---

## Underutilized Patterns

### Pattern: Low Occupancy — Register Limited
**Metrics**: `launch__occupancy_limit_registers` < `launch__occupancy_limit_blocks`
**Likely cause**: Kernel uses too many registers per thread
**Code to check, by framework**:

| Framework | Pattern to look for |
|-----------|-------------------|
| Triton | `num_stages` too high (each stage adds registers for prefetch buffers); complex expressions with many live variables; large `BLOCK_SIZE` |
| CUDA (Numba) | Too many local variables; large loop bodies; unrolled loops |
| TileLang | `T.Pipelined(num_stages=...)` too many stages; `T.alloc_fragment` too large |

### Pattern: Low Occupancy — Shared Memory Limited
**Metrics**: `launch__occupancy_limit_shared_mem` < other limits
**Likely cause**: Too much shared memory per block
**Code to check, by framework**:

| Framework | Pattern to look for |
|-----------|-------------------|
| Triton | `num_stages` — each pipeline stage allocates shared memory buffers; large tile dimensions |
| CUDA (Numba) | Large `cuda.shared.array()` allocations |
| TileLang | `T.alloc_shared` with large sizes; `T.Pipelined` with many stages |

### Pattern: High Long Scoreboard Stall + Low Occupancy
**Metrics**: `smsp__warp_issue_stalled_long_scoreboard` > 30%, occupancy limited
**Likely cause**: Not enough warps to hide DRAM latency
**Code to check**: Increase `num_warps`/threads per block; increase pipeline stages for more prefetch overlap (watch register/shared mem trade-off).

### Pattern: High Barrier Stall
**Metrics**: `smsp__warp_issue_stalled_barrier` > 15%
**Likely cause**: Synchronization overhead from reductions or explicit barriers
**Code to check**: All reduce/sum/max operations require warp/block synchronization. Softmax normalization involves multiple reductions. Check if reductions can be fused or the reduction dimension restructured.

### Pattern: High Memory Dependency Stall
**Metrics**: `smsp__warp_issue_stalled_memory_dependency` > 25%
**Likely cause**: Warps blocked waiting for memory operands to become available
**Code to check**: Chain of dependent loads; index-based lookups creating dependent memory access chains (common in sparse operations); insufficient software pipelining.

### Pattern: High Short Scoreboard Stall
**Metrics**: `smsp__warp_issue_stalled_short_scoreboard` > 20%
**Likely cause**: Shared memory or L1 cache contention
**Code to check**: Shared memory bank conflicts — access patterns where multiple threads hit the same bank. Swizzling can help with L2 locality but doesn't directly fix shared memory banks.

---

## Sparse / Irregular Access Patterns

### Index Lookup Overhead
**Signature**: High memory dependency stall + extra DRAM reads beyond Q/K/V
**Cause**: Loading sparsity patterns (CSR row_ptr, col_idx, block indices) creates dependent access chains
**Check**: Count total `dram__bytes_read` vs theoretical Q+K+V+Output bytes — excess is index overhead

### Irregular Block Sizes
**Signature**: Low coalescing + warp divergence + underutilization
**Cause**: Variable-length sparse blocks cause different warps to do different amounts of work
**Check**: Branch uniformity metric — if low, warps are diverging due to mask/boundary conditions

### Reduction Over Sparse Blocks
**Signature**: High barrier stall + moderate compute
**Cause**: Softmax normalization over variable-number-of-blocks requires careful reduction
**Check**: Whether the kernel does online softmax (streaming) vs two-pass softmax — online is generally better for sparse
