# NSYS Metrics Reference — Timeline Analysis Guide

## Key Concepts

### GPU Utilization
- **GPU Busy %**: Fraction of wall time where at least one kernel is executing on the GPU
- Target: >90% for well-optimized workloads
- <50% usually means CPU bottleneck, excessive sync, or launch overhead

### Kernel Launch Overhead
- Each kernel launch has fixed CPU-side overhead (~5-15us per launch)
- For kernels shorter than ~50us, launch overhead becomes significant
- JIT-compiled frameworks (Triton, Numba CUDA, TileLang) add extra latency on first launch (warmup)

### Memory Transfer Categories
| copyKind | Direction | Description |
|----------|-----------|-------------|
| 1 | HtoD | Host to Device (CPU->GPU memory) |
| 2 | DtoH | Device to Host (GPU->CPU memory) |
| 8 | DtoD | Device to Device (GPU internal) |
| 10 | HtoD (Pinned) | Pinned host to device (faster) |

---

## CUDA API Calls — What to Watch

### High-Impact APIs

| API | Normal | Warning Sign |
|-----|--------|--------------|
| `cudaDeviceSynchronize` | Occasional, fast | Many calls or long total time — CPU blocking on GPU |
| `cudaLaunchKernel` | Per-kernel, ~5-15us | If total launch time > kernel compute time |
| `cudaMalloc` | During init only | Calls during kernel loop = allocation overhead |
| `cudaFree` | During cleanup | Calls during kernel loop = deallocation overhead |
| `cudaMemcpy` | For data transfer | Unexpected copies during compute phase |
| `cudaStreamSynchronize` | Per-stream sync | Better than DeviceSync if using streams |

### JIT Compilation APIs
- `cuModuleLoadData` / `cuModuleGetFunction`: JIT compilation and module loading — should only appear during warmup
- `cuLaunchKernel`: The actual kernel launch (all GPU DSLs ultimately use CUDA driver API)
- `nvrtcCreateProgram` / `nvrtcCompileProgram`: Runtime compilation (Numba CUDA)

---

## Timeline Patterns

### Pattern: Low GPU Utilization Despite Many Kernels
**Symptom**: GPU busy < 60%, many kernel launches
**Cause**: Kernel launch overhead dominates
**Check**: Compare total kernel time vs total wall time. If ratio is low, launch overhead is the issue.
**Action**: For small kernels, consider fusing operations.

### Pattern: Long cudaDeviceSynchronize
**Symptom**: `cudaDeviceSynchronize` appears in top API calls with high total time
**Cause**: CPU blocks waiting for GPU to finish all pending work
**Check**: Is `torch.cuda.synchronize()` called in the profiled loop?
**Action**: torch.cuda.synchronize() in the benchmark wrapper is expected, but it shouldn't appear inside the kernel launch loop.

### Pattern: Large H2D Transfers in Kernel Loop
**Symptom**: Many HtoD memcpy operations during profiling
**Cause**: Data being copied to GPU on each iteration instead of pre-staging
**Check**: Are input tensors on GPU before the loop? Is `.cuda()` called inside the loop?
**Action**: Ensure all input tensors are on CUDA before the profiled section.

### Pattern: First Kernel Much Slower
**Symptom**: Max kernel time >> Avg kernel time (e.g., 10x or more)
**Cause**: JIT compilation on first invocation (all GPU DSLs)
**Check**: Compare first vs subsequent kernel durations
**Action**: This is expected. The profiling script should include warmup iterations.

### Pattern: DtoH Transfers After Each Kernel
**Symptom**: DtoH memcpy interleaved with kernel launches
**Cause**: Results being copied back to CPU after each kernel call
**Check**: Is the code doing `.cpu()` or `.item()` on outputs in the loop?
**Action**: Avoid reading results back to CPU during profiling.

---

## Interpreting Kernel Names

GPU DSLs generate mangled kernel names:

| Framework | Name Pattern | Example |
|-----------|-------------|---------|
| Triton | `<module>_<func>_<hash>` | `triton_poi_fused_add_mul_0` |
| Numba CUDA | `cudapy::__main__::<func>` | `cudapy::__main__::my_kernel` |
| TileLang | `<func>_kernel0` | `main_kernel0` |
| Raw CUDA | `<func_name>` | `vector_add` |

Tips:
- Sort by total time to find the dominant kernel
- Multiple kernel variants may appear if autotune is used (autotuner tries different configs during warmup)
- The function name from your source code usually appears in the mangled name

---

## Quick Diagnostics Checklist

1. **GPU utilization < 50%?** -> Check for CPU bottlenecks, excessive sync, or launch overhead
2. **Total memcpy time > 10% of kernel time?** -> Data transfer optimization needed
3. **First kernel > 10x average?** -> Warmup issue, need more warmup iterations
4. **cudaDeviceSynchronize in hot path?** -> Remove unnecessary sync points
5. **Many tiny kernels (<10us each)?** -> Consider kernel fusion
6. **High DtoD memcpy?** -> Check for unnecessary tensor copies on GPU
