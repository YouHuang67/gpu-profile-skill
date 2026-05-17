---
name: ncu-profile
description: "/ncu-profile: NCU hardware bottleneck analysis. Triggers: ncu, hardware counter, roofline, bottleneck, occupancy, stall analysis, memory bandwidth, tensor core, coalescing, profile kernel."
triggers:
  - /ncu-profile
  - ncu profile
  - hardware counter
  - profile this kernel
  - analyze bottleneck
  - roofline analysis
  - check occupancy
  - stall analysis
  - memory bandwidth analysis
  - tensor core utilization
  - coalescing check
---

# NCU Profile

Run NCU profiling on a GPU kernel, interpret hardware metrics, present bottleneck analysis.

**Rule**: Only profile when user explicitly asks. Do not suggest unprompted.

## Common Blockers

### Sudo / GPU Permission
```bash
cat /proc/driver/nvidia/params 2>/dev/null | grep RmProfilingAdminOnly
```
| Output | Action |
|--------|--------|
| `RmProfilingAdminOnly: 0` | Proceed |
| `RmProfilingAdminOnly: 1` | Script auto-uses sudo |
| File unreadable | Assume sudo needed |

Permanent fix (reboot required):
```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u
sudo reboot
```

### NCU Binary
Auto-search: `$CUDA_HOME/bin/ncu` -> PATH -> `/usr/local/cuda/bin/ncu` -> `/usr/local/cuda-*/bin/ncu`.

```bash
python -c "
import shutil, os, glob
ncu = shutil.which('ncu')
if ncu: print('NCU on PATH:', ncu)
for p in sorted(glob.glob('/usr/local/cuda*/bin/ncu')):
    print('Found:', p)
"
```

If not found: install [Nsight Compute](https://developer.nvidia.com/nsight-compute) or pass `--ncu /path/to/ncu`.

## Step 1: Environment Check
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
```

## Step 2: Detect Entry Point
```bash
python scripts/detect_entry.py "$ARGUMENTS" --json
```

| Result | Meaning | Action |
|--------|---------|--------|
| `error` | File issue | Report, stop |
| `main_block` | Runnable | Use as-is |
| `callable` | Has launch functions | Generate wrapper, confirm inputs |
| `kernel_only` | Only kernels, or JIT/pre-compiled (no Python decorator) | NCU captures all CUDA launches. Proceed directly. |

## Step 3: Run NCU
```bash
python scripts/run_ncu.py \
    <script> \
    -o /tmp/gpu_profile_ncu_$(date +%s) \
    --python "$(which python)" \
    --timeout 300
```

Options:
- Explicit NCU: `--ncu /usr/local/cuda-12.4/bin/ncu`
- NVTX skip warmup: `--nvtx-include "profile_target"`
- Kernel filter: `-k "kernel_name"` (for JIT use short substring, names include template params like `void kernel<(int)64,...>`)
- Launch control `-s N -n M` are GLOBAL counts. PyTorch internal launches also count. Prefer NVTX over `-s`/`-n` for JIT.
- Detail level: `--page details` for per-metric text (complete), default `--page raw` for CSV summary (fast).

## Step 4: Error Handling
| Symptom | Cause | Fix |
|---------|-------|-----|
| Permission denied | RmProfilingAdminOnly=1 | Use sudo or permanent fix |
| No .ncu-rep | No kernel launched | Verify script runs GPU kernels |
| "No kernels profiled" with `-k` | Regex didn't match JIT name | Drop `-k`, profile all, filter in results |
| `-s`/`-n` skip target | Global count includes PyTorch ops | Don't use `-s`/`-n` for JIT. Use NVTX. |
| Timed out | Too slow | `--timeout 600` or reduce data |
| Module not found | Wrong env | Verify Python env, deps |
| exit 137 | OOM | Reduce tensor sizes |

## Step 5: Analyze Results

When `--page raw` gives only section summaries, re-run with `--page details` for per-metric breakdown.

### 1. Roofline
- Memory Bound: memory_sol >= compute_sol
- Compute Bound: compute_sol > memory_sol
- Underutilized: both < 60%

### 2. Metric Groups (10 groups)
Interpret with `reference/ncu_metrics.md`.
1. Speed of Light (SOL)  6. Global Memory Access
2. SM & Compute           7. Occupancy & Resources
3. DRAM (HBM)           8. Launch Configuration
4. L2 Cache              9. Stall Analysis
5. L1/TEX Cache         10. Timing

### 3. Derived Metrics
- Global Load/Store Sectors/Request (1=perfect, >4=waste, 32=worst)
- L2 Hit Rate, DRAM Bandwidth (GB/s), Kernel Duration (ms)

### 4. Code-Level Mapping
Map to source with `reference/bottleneck_patterns.md`. State confidence: certain / likely / speculative.

### 5. Key Findings
Top 3-5 by impact. Each: What + Evidence + Where + Impact (High/Medium/Low).

**Do NOT suggest fixes unless asked.**

## Follow-Up
- "Open `.ncu-rep` in GUI?" (give path)
- "Re-run with `--page details` for full breakdown?"
- "Run `/nsys-profile` for timeline data?"
