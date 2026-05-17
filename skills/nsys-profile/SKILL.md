---
name: nsys-profile
description: "/nsys-profile: NSYS timeline analysis. Triggers: nsys, timeline, gpu utilization, kernel timing, launch overhead, memory transfer, cpu gpu overlap, kernel fusion, trace."
triggers:
  - /nsys-profile
  - nsys profile
  - timeline analysis
  - gpu utilization
  - kernel timing
  - launch overhead
  - memory transfer analysis
  - cpu gpu overlap
---

# NSYS Profile — GPU Kernel Timeline & Utilization Analysis

Run NSYS profiling on a GPU kernel, analyze the timeline, and present GPU utilization, kernel timing, and overhead findings.

**Rule**: Only run profiling when the user explicitly asks. Do NOT suggest it unprompted.

## Common Blocker: NSYS Binary

Auto-search: `$CUDA_HOME/bin/nsys` → PATH → `/usr/local/cuda/bin/nsys` → `/usr/local/cuda-{12.8..11.8}/bin/nsys`.
NSYS usually does NOT need sudo (unlike NCU).

```bash
python -c "
import shutil, os
nsys = shutil.which('nsys')
if nsys: print('NSYS on PATH:', nsys)
for v in ['12.8','12.6','12.4','12.2','12.0','11.8']:
    p = f'/usr/local/cuda-{v}/bin/nsys'
    if os.path.isfile(p): print('Found:', p)
"
```

If not found: install [Nsight Systems](https://developer.nvidia.com/nsight-systems).
If permission fails: `cat /proc/sys/kernel/perf_event_paranoid` (needs <= 2).

## Step 1: Environment Check

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

## Step 2: Detect Entry Point

```bash
python ${CLAUDE_SKILL_DIR}/../gpu-scripts/detect_entry.py "$ARGUMENTS" --json
```

| Result | Action |
|--------|--------|
| `error` | Report, stop |
| `main_block` | Use as-is |
| `callable` | Generate wrapper, confirm inputs |
| `kernel_only` | Ask user how to invoke |

## Step 3: Run NSYS

```bash
python ${CLAUDE_SKILL_DIR}/../gpu-scripts/run_nsys.py \
    <script> \
    -o /tmp/gpu_profile_nsys_$(date +%s) \
    --python "$(which python)" \
    --timeout 120
```

Explicit NSYS path: `--nsys /usr/local/cuda-12.4/bin/nsys`

Output: `.nsys-rep` + JSON (stats + SQLite summary) to stdout.

## Step 4: Error Handling

| Symptom | Cause | Fix |
|---------|-------|-----|
| Permission denied | perf_event_paranoid > 2 | `sudo sysctl kernel.perf_event_paranoid=2` |
| No .nsys-rep | No CUDA kernel | Verify script launches GPU kernels |
| Timed out | Too long | Increase `--timeout` |
| Module not found | Wrong Python env | Check env, dependencies |

## Step 5: Analyze Results

Read the JSON output. Present in this structure:

### 1. Overall GPU Utilization

From SQLite summary (if available):
- **Wall time**: total profiled duration
- **GPU utilization**: % of time GPU was executing kernels
- **GPU idle time**: time between kernels where GPU was idle

High idle time → kernel launch overhead or CPU-side bottleneck.

### 2. Kernel Execution Summary

| Kernel Name | Count | Total (us) | Avg (us) | Min (us) | Max (us) |

- One dominant kernel or many small ones?
- High variance (max >> avg) → data-dependent execution
- Many small invocations → launch overhead dominates

### 3. Memory Transfer Analysis

| Direction | Count | Total Bytes | Total Time (us) |
|-----------|-------|-------------|-----------------|

- H2D/D2H ratio and volume
- Transfers during compute (overlap) or blocking?
- Repeated H2D for same data?

### 4. CUDA API Overhead

Top API calls by total time. Key signals:
- `cudaDeviceSynchronize` — CPU blocking on GPU
- `cudaLaunchKernel` — launch overhead
- `cudaMalloc`/`cudaFree` — should only appear during init/cleanup

### 5. Timeline Patterns

1. **Launch overhead**: total launch time vs compute time
2. **Sync bottlenecks**: time in sync APIs
3. **Memory transfer overlap**: concurrent or serial?
4. **Warmup effects**: first invocation much slower (JIT compilation)

### 6. Key Findings

Top 3-5 findings, each with:
1. Finding
2. Evidence (specific timing values)
3. What to investigate further

## Follow-Up

- "Open `.nsys-rep` in Nsight Systems GUI?" (give path)
- "Run `/ncu-profile` on the dominant kernel for hardware-level bottleneck analysis?"

Reference: `${CLAUDE_SKILL_DIR}/../gpu-refs/nsys_metrics.md`
