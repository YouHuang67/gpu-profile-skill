# NCU Metrics Reference

## Group 1: Speed of Light (SOL)

| Metric | Description | Unit | Interpretation |
|--------|-------------|------|----------------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | **Compute SOL** — SM throughput | % | >60% = compute bound |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | **Memory SOL** — memory subsystem throughput | % | >60% = memory bound |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | DRAM throughput SOL | % | Alternative memory SOL |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1/TEX throughput SOL | % | High = heavy cache usage |

### Roofline Decision
- Both SOL < 60% → **Underutilized** (occupancy/stalls/launch config)
- Memory SOL >= Compute SOL → **Memory Bound**
- Compute SOL > Memory SOL → **Compute Bound**

---

## Group 2: SM & Compute Utilization

| Metric | Description | Unit | Good Range |
|--------|-------------|------|------------|
| `sm__cycles_active.avg` | Average cycles SM was active | cycles | Higher = more work |
| `sm__cycles_elapsed.avg` | Total elapsed SM cycles | cycles | Active/Elapsed = SM busy ratio |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy | % | >50% good, <25% problem |
| `sm__warps_active.avg.per_cycle_active` | Active warps per active cycle | warps | Higher = better latency hiding |
| `sm__inst_executed.sum` | Total instructions executed | inst | Context-dependent |
| `sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active` | FP32 pipe utilization | % | High if FP32 compute |
| `sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active` | Tensor core pipe utilization | % | Should be high for GEMM |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | Tensor core active ratio | % | >5% = TC in use |
| `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active` | Load/Store unit utilization | % | High = memory-intensive |

### Interpretation
- Tensor pipe near 0 with matrix ops → TC not used (check dtype: need fp16/bf16)
- Low warps_active + low SOL → occupancy problem
- High LSU utilization → kernel is memory-operation heavy

---

## Group 3: DRAM (HBM)

| Metric | Description | Unit | Good Range |
|--------|-------------|------|------------|
| `dram__bytes.sum` | Total DRAM bytes transferred | bytes | Compare with theoretical min |
| `dram__bytes_read.sum` | Bytes read from DRAM | bytes | |
| `dram__bytes_write.sum` | Bytes written to DRAM | bytes | |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM throughput % | % | >80% = near peak BW |
| `dram__bytes.sum.per_second` | DRAM bandwidth | B/s | Compare with GPU peak |
| `dram__sectors_read.sum` | DRAM sectors read | sectors | For efficiency calculation |
| `dram__sectors_write.sum` | DRAM sectors written | sectors | For efficiency calculation |

### Interpretation
- Compare actual bytes with theoretical minimum (input + output sizes) → ratio is memory efficiency
- High DRAM throughput near peak is expected for bandwidth-bound kernels and may be optimal

---

## Group 4: L2 Cache

| Metric | Description | Unit | Good Range |
|--------|-------------|------|------------|
| `lts__t_sectors_lookup_hit.sum` | L2 sectors hit | sectors | For computed hit rate |
| `lts__t_sectors_lookup_miss.sum` | L2 sectors missed | sectors | High miss = thrashing |
| `lts__t_sector_hit_rate.pct` | L2 hit rate | % | >50% for data reuse |
| `lts__t_sectors.sum` | Total L2 sectors accessed | sectors | |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | L2 throughput SOL | % | |

### Interpretation
- L2 hit rate < 20% → data exceeds L2 capacity or no reuse (e.g., streaming access)
- High L2 throughput + low hit rate = L2 thrashing

---

## Group 5: L1/TEX Cache

| Metric | Description | Unit | Good Range |
|--------|-------------|------|------------|
| `l1tex__t_sectors_lookup_hit.sum` | L1 sectors hit | sectors | |
| `l1tex__t_sectors_lookup_miss.sum` | L1 sectors missed | sectors | |
| `l1tex__t_sector_hit_rate.pct` | L1 hit rate | % | >80% good, <50% problem |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum` | Shared memory wavefronts | wavefronts | High = heavy shared mem use |

### Interpretation
- Low L1 hit → poor spatial locality in global memory access patterns
- Shared memory wavefronts indicate shared memory traffic volume

---

## Group 6: Global Memory Access

| Metric | Description | Unit | Good Range |
|--------|-------------|------|------------|
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | Global load sectors | sectors | For sectors/request |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` | Global store sectors | sectors | For sectors/request |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | Global load requests | requests | |
| `l1tex__t_requests_pipe_lsu_mem_global_op_st.sum` | Global store requests | requests | |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | Load coalescing efficiency | % | 100% = perfect |
| `smsp__sass_average_branch_targets_threads_uniform.pct` | Branch uniformity | % | 100% = no divergence |

### Derived: Sectors/Request
- **sectors/request = sectors / requests**
- **1.0** = perfectly coalesced (ideal)
- **>4.0** = significant coalescing waste
- **32.0** = worst case (every thread in warp accesses different cache line)
- This is one of the most actionable metrics for GPU kernels

---

## Group 7: Occupancy & Resources

| Metric | Description | Unit | Interpretation |
|--------|-------------|------|----------------|
| `launch__occupancy_limit_blocks` | Occupancy limit: block count | % | GPU max |
| `launch__occupancy_limit_registers` | Occupancy limit: registers | % | Lower = register pressure |
| `launch__occupancy_limit_shared_mem` | Occupancy limit: shared mem | % | Lower = shared mem pressure |
| `launch__occupancy_limit_warps` | Occupancy limit: warps | % | |
| `launch__occupancy_per_block_size` | Theoretical occ (block size) | % | |
| `launch__occupancy_per_register_count` | Theoretical occ (registers) | % | |
| `launch__occupancy_per_shared_mem_size` | Theoretical occ (shared mem) | % | |
| `smsp__warps_active.avg.pct_of_peak_sustained_active` | Warp occupancy % of peak | % | >50% adequate |

### Interpretation
- The occupancy limiter is the minimum of all limits
- Register-limited: reduce pipeline stages, simplify kernel, adjust warps/threads
- Shared-mem-limited: reduce tile size or pipeline depth (more stages = more shared mem buffers)
- Software pipelining (num_stages in Triton, Pipelined in TileLang) directly controls shared memory usage

---

## Group 8: Launch Configuration

| Metric | Description | Unit |
|--------|-------------|------|
| `launch__block_size` | Threads per block | threads |
| `launch__grid_size` | Total blocks launched | blocks |
| `launch__registers_per_thread` | Registers per thread | regs (>128 is high) |
| `launch__shared_mem_per_block_dynamic` | Dynamic shared memory | bytes |
| `launch__shared_mem_per_block_static` | Static shared memory | bytes |
| `launch__waves_per_multiprocessor` | Waves per SM | waves (<1.0 = not enough work) |

### Interpretation
- waves_per_multiprocessor < 1.0 → grid is too small to fill the GPU
- High registers_per_thread → limits occupancy
- Total shared mem = dynamic + static, compare with GPU max per SM

---

## Group 9: Stall Analysis

| Metric | Description | Unit | Threshold |
|--------|-------------|------|-----------|
| `smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct` | Memory operand wait | % | >20% = latency issue |
| `smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct` | L1/shared/tex latency | % | >15% = L1/shared contention |
| `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` | L2/DRAM latency | % | >20% = latency not hidden |
| `smsp__warp_issue_stalled_barrier_per_warp_active.pct` | Barrier sync wait | % | >10% = sync overhead |
| `smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct` | Branch resolution | % | >5% = divergence |

### Stall Priority (focus on largest first)
1. Long scoreboard + memory dependency → memory latency (most common)
2. Short scoreboard → shared memory / L1 issues
3. Barrier → sync/reduction overhead
4. Branch → divergence

---

## Group 10: Timing

| Metric | Description | Unit |
|--------|-------------|------|
| `gpu__time_duration.sum` | Kernel wall-clock duration | ns |
| `gpu__time_active.sum` | GPU active time | ns |

### Derived Metrics
- **kernel_duration_ms** = duration / 1e6
- **dram_bandwidth_gbps** = dram_bytes / duration_ns


---

## GPU Spec Reference

When interpreting NCU metrics, compare against the GPU's hardware limits. Query with:

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
```

Common reference values (verify for your specific GPU):

| GPU | Arch | Peak DRAM BW | Peak FP16 TFLOPS | SMs | L2 Cache |
|-----|------|-------------|------------------|-----|----------|
| RTX 3090 | Ampere (8.6) | 936 GB/s | 35.6 | 82 | 6 MB |
| A100 | Ampere (8.0) | 1555 GB/s | 312 | 108 | 40 MB |
| H100 | Hopper (9.0) | 2039 GB/s | 990 | 132 | 50 MB |
| RTX 4090 | Ada (8.9) | 1008 GB/s | 82.6 | 128 | 72 MB |

For other GPUs, look up specs from NVIDIA documentation or compute the peak DRAM bandwidth from the memory bus width and clock rate.

## NCU Command Quick Reference

Common NCU commands for reference (all wrapped by `run_ncu.py`):

```bash
# Basic: profile all kernels, CSV output
ncu --set full --target-processes all --export report.ncu-rep python script.py

# Filter: only specific kernel name (regex)
ncu --kernel-name "my_kernel" --launch-skip 5 --launch-count 3 ...

# NVTX: only profile code inside nvtx range
ncu --nvtx --nvtx-include "profile_target/" ...

# Export: binary report for GUI
ncu --export report.ncu-rep ...

# Import: extract data from report
ncu --import report.ncu-rep --csv --page raw      # section summary
ncu --import report.ncu-rep --page details        # full per-metric text
```

Notes for run_ncu.py users:
- `--launch-skip (-s)` and `--launch-count (-n)` count ALL CUDA kernel launches globally
- In mixed workloads (e.g., PyTorch + custom JIT kernel), PyTorch internal ops count toward skip/count
- For JIT kernels where you cannot predict launch count, omit `-s`/`-n` and use `--nvtx-include`
- For pre-compiled .so files, kernel names are from the compiled binary; profile without `-k` first to discover names
