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

# NSYS Profile

Run NSYS profiling on a GPU kernel, analyze timeline, present utilization and timing findings.

**Rule**: Only profile when user explicitly asks.

## Common Blocker: NSYS Binary
Auto-search: `$CUDA_HOME/bin/nsys` -> PATH -> `/usr/local/cuda/bin/nsys` -> `/usr/local/cuda-*/bin/nsys`.
NSYS usually does NOT need sudo.

```bash
python -c "
import shutil, os, glob
nsys = shutil.which('nsys')
if nsys: print('NSYS on PATH:', nsys)
for p in sorted(glob.glob('/usr/local/cuda*/bin/nsys')):
    print('Found:', p)
"
```

If not found: install [Nsight Systems](https://developer.nvidia.com/nsight-systems).
If permission fails: `cat /proc/sys/kernel/perf_event_paranoid` (needs <= 2).

## Step 1: Environment
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

## Step 2: Entry Point
```bash
python scripts/detect_entry.py "$ARGUMENTS" --json
```

## Step 3: Run NSYS
```bash
python scripts/run_nsys.py \
    <script> \
    -o /tmp/gpu_profile_nsys_$(date +%s) \
    --python "$(which python)" \
    --timeout 120
```

Explicit path: `--nsys /usr/local/cuda-12.4/bin/nsys`

## Step 4: Error Handling
| Symptom | Cause | Fix |
|---------|-------|-----|
| Permission denied | perf_event_paranoid > 2 | `sudo sysctl kernel.perf_event_paranoid=2` |
| No .nsys-rep | No kernel | Verify script launches GPU kernels |
| Timed out | Too long | Increase `--timeout` |

## Step 5: Results

### 1. GPU Utilization
Wall time, GPU busy %, idle time.

### 2. Kernel Summary
| Kernel Name | Count | Total (us) | Avg (us) | Min (us) | Max (us) |

### 3. Memory Transfer
| Direction | Count | Total Bytes | Total Time (us) |

### 4. CUDA API Overhead
Watch: `cudaDeviceSynchronize`, `cudaLaunchKernel`, `cudaMalloc`/`cudaFree`.

### 5. Timeline Patterns
- Launch overhead: total launch vs compute
- Sync bottlenecks
- Memory overlap
- Warmup effects (JIT compilation)

### 6. Key Findings
Top 3-5, each: finding + evidence + investigate.

## Reference
`reference/nsys_metrics.md` (includes SQLite schema quick-ref).
