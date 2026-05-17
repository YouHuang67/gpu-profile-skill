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


---

## Analysis Workflow Guide

These are reference patterns. Adapt to the specific kernel and metrics at hand.

### Step 1: Read the Kernel Source First

Before looking at any numbers, understand the kernel:
- What operation? (matmul, elementwise, reduction, attention, convolution...)
- Data types (fp16, bf16, fp32, int8...)
- Tile sizes, block dimensions, warp counts, pipeline stages
- Memory access patterns (contiguous, strided, gather, scatter)

This tells you what values to expect. A matmul should show tensor core activity. A streaming kernel should have low L1 hit rate and high DRAM throughput. If the numbers don't match the operation type, something is wrong.

### Step 2: Roofline First

Start every report with the roofline classification. It frames everything that follows.

| Metric | Meaning |
|--------|---------|
| `gpu__compute_memory_throughput` (Memory SOL) | How close to peak memory bandwidth |
| `sm__throughput` (Compute SOL) | How close to peak compute throughput |
| Both < 60% | Underutilized. The GPU has spare capacity. Look at occupancy and stalls. |

### Step 3: Diagnose by Classification

**Memory-bound** (Memory SOL >= Compute SOL)
- Check coalescing: Sectors/Request. 1.0 = perfect, >4 = waste, >32 = every thread hits different cache line
- Check L1/L2 hit rates. Low L1 + high L2 = streaming access (may be intentional). Low L2 = data exceeds cache.
- Check DRAM throughput vs GPU peak

**Compute-bound** (Compute SOL > Memory SOL)
- Check tensor core activity. <5% on matmul = TCs not engaged (likely dtype issue)
- Check FP32 pipe. High on fp16 workload = dtype mismatch somewhere

**Underutilized** (both < 60%)
- Check occupancy limiter. Which resource is the bottleneck: registers, shared memory, or warps?
- Check Waves/SM. <1 means not enough work. >10 means too many tiny blocks (launch overhead).
- Check stall reasons. Math pipe stall + low eligible warps = not enough warps to hide latency. Long scoreboard = DRAM latency dominating.
- 255 regs/thread means the compiler spilled nothing but the register file is full; only a few warps can run per SM. 128 regs on a 4-warp kernel is expected. >128 on a 4-warp kernel means occupancy will be limited.

### Step 4: Explain Metrics as You Go

When presenting a metric, always include what it means and what the number signifies. Examples:

Good (explains meaning):
```
dkv Active Warps/Scheduler: 1.93 out of 4 possible. This SM has 4 warp schedulers
but only ~2 warps available to pick from. When a warp stalls on a math instruction,
there's often no other warp ready, so the scheduler idles.
```

Bad (just reports value):
```
dkv Active Warps/Scheduler: 1.93
```

Good:
```
Waves/SM: 50. This means 50 blocks are queued for each SM. The grid is 8192 blocks
on 82 SMs. Each block runs fast, but the GPU spends significant time switching
between blocks. Compare to a well-tuned kernel where Waves/SM is 2-4.
```

Bad:
```
Waves/SM: 49.95
```

Good:
```
Store coalescing: 2.0 bytes per sector out of 32 possible. Only 6.3% of each
64-byte memory transaction carries useful data. The remaining 93.7% is wasted
bandwidth. This happens when threads in a warp write to non-contiguous addresses.
```

### Step 5: Map to Code

For each finding, point to the specific code location and explain the causal chain:

```
Finding: dq smem limits occupancy to 1 block per SM
  Evidence: Block Limit Shared Mem = 1, Occupancy = 8%, Active Warps = 1.00
  Location: det_dq.cuh lines 80-89, SharedStorage struct
  Why: Q/K/V tiles total 65KB (>48KB SM limit), so only 1 block fits per SM.
       With 1 block of 4 warps, at most 1 warp runs per scheduler.
  Confidence: certain
```

### Step 6: End with a Focused Conclusion

After the detailed sections, provide a short, direct conclusion:

1. State the primary bottleneck in one sentence
2. Connect it to the evidence (1-2 key numbers)
3. If multiple findings, show the causal chain (e.g., "X causes Y which results in Z")
4. Compare to a baseline if available (e.g., "1.37x slower than FA2")
5. One concrete direction (not a fix, just what to look at)

Example:
```
Bottom line: Wave explosion is the root cause. Grid of 8192 blocks on 82 SMs
creates 50-100 waves per SM. Most of the GPU's time is spent switching between
blocks, not computing. Within each block, only 1-2 warps are available per
scheduler due to 255 regs/thread and 65KB smem, so math pipe latency cannot
be hidden. The store coalescing issue (6.3%) and L1 bypass are secondary;
they only matter after occupancy is fixed. Compared to FA2's 64 blocks on
82 SMs (<1 wave/SM), the scheduling overhead difference is clear.
```

### Typical Metric Ranges

Reference ranges for common GPU kernels. Results vary by GPU and workload.

| Category | Metric | Good | Warning | Critical |
|----------|--------|------|---------|----------|
| Coalescing | Load Sectors/Request | 1.0-2.0 | 2.0-4.0 | >4.0 |
| L1 Cache | Hit Rate | >80% | 50-80% | <50% |
| L2 Cache | Hit Rate | >50% | 20-50% | <20% |
| Occupancy | Warps Active % | >50% | 25-50% | <25% |
| Stall | Long Scoreboard | <20% | 20-35% | >35% |
| Stall | Barrier | <10% | 10-20% | >20% |
| Stall | Memory Dependency | <20% | 20-30% | >30% |
| Registers | Per Thread | <64 | 64-128 | >128 |
| Waves | Per SM | 1-4 | 4-10 | >10 or <1 |

