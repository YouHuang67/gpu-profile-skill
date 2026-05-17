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

# NCU Profile — GPU Kernel Hardware Bottleneck Analysis

Run NCU profiling on a GPU kernel, interpret hardware metrics, and present a clear bottleneck analysis.

**Rule**: Only run profiling when the user explicitly asks. Do NOT suggest it unprompted.
When a file path is unclear, ask the user to specify.

## Common Blockers (check before profiling)

### Sudo / GPU Permission

NCU reads hardware performance counters. Check permission:

```bash
cat /proc/driver/nvidia/params 2>/dev/null | grep RmProfilingAdminOnly
```

| Output | Meaning | Action |
|--------|---------|--------|
| `RmProfilingAdminOnly: 0` | No restriction | Proceed |
| `RmProfilingAdminOnly: 1` | Root-only | Script auto-uses `sudo ncu`; user may need to enter password |
| File unreadable | Unknown | Assume sudo needed |

Permanent fix (requires reboot):
```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u && sudo reboot
```

### NCU Binary Location

Auto-search order: `$CUDA_HOME/bin/ncu` -> PATH -> `/usr/local/cuda/bin/ncu` -> `/usr/local/cuda-{12.8..11.8}/bin/ncu`.

```bash
python -c "
import shutil, os
ncu = shutil.which('ncu')
if ncu: print('NCU on PATH:', ncu)
for v in ['12.8','12.6','12.4','12.2','12.0','11.8']:
    p = f'/usr/local/cuda-{v}/bin/ncu'
    if os.path.isfile(p): print('Found:', p)
"
```

If not found: install [Nsight Compute](https://developer.nvidia.com/nsight-compute) or pass `--ncu /path/to/ncu`.

## Step 1: Environment Check

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); d=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'; print('Device:', d)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
```

## Step 2: Detect Entry Point

```bash
python ${CLAUDE_SKILL_DIR}/../gpu-scripts/detect_entry.py "$ARGUMENTS" --json
```

| Result | Meaning | Action |
|--------|---------|--------|
| `error` | File issue | Report error, stop |
| `main_block` | Runnable script | Use as-is |
| `callable` | Has launch functions, no `__main__` | Generate wrapper at `/tmp/gpu_profile_*/wrapper.py`, confirm inputs |
| `kernel_only` | Only kernel definitions, or no Python decorators (JIT/C++ kernel) | For JIT: NCU captures all CUDA launches regardless; proceed. Kernel name filters may need to match mangled/template names. |

## Step 3: Run NCU

```bash
python ${CLAUDE_SKILL_DIR}/../gpu-scripts/run_ncu.py \
    <script> \
    -o /tmp/gpu_profile_ncu_$(date +%s) \
    --python "$(which python)" \
    --timeout 300
```

Options:
- Explicit NCU path: `--ncu /usr/local/cuda-12.4/bin/ncu`
- NVTX (skip autotune warmup): `--nvtx-include "profile_target"`
  The test script must wrap the target with: `with torch.cuda.nvtx.range("profile_target"): kernel(...)`
- Kernel filter: `-k "kernel_name"` — regex matches kernel names. For JIT kernels, names include template params like `void det_attn_bwd_dkv_kernel<(int)64,...>`. Use a short substring match (e.g. `-k "det_attn_bwd"`).
- Launch control: `-s N -n M` — skip N global launches, profile M. These are GLOBAL counts: in mixed PyTorch+JIT workloads, PyTorch internal launches also count. Prefer NVTX over `-s`/`-n` for JIT.
- Detail level: `--page details` for per-metric text output (slower but complete). Default is `--page raw` (CSV section summary, fast).

Output: `.ncu-rep` (GUI-openable) + metrics JSON to stdout.

## Step 4: Error Handling

| Symptom | Cause | Fix |
|---------|-------|-----|
| Permission denied | RmProfilingAdminOnly=1 | Use sudo or permanent fix |
| No .ncu-rep produced | No CUDA kernel launched | Verify script runs GPU kernels |
| "No kernels were profiled" with `-k` | Regex didn't match JIT kernel name | Drop `-k`, run without filter, then filter by name in results. Use `--kernel-name-filter` in post-processing instead. |
| Timed out (300s) | Profiling too slow | `--timeout 600` or reduce data |
| Module not found | Wrong Python env | Verify Python env, dependencies |
| exit 137 (OOM) | Too much memory | Reduce input tensor sizes |
| ncu: command not found | NCU not installed | Install or `--ncu` flag |
| `-s`/`-n` skip target kernel | Global launch count includes PyTorch internal ops | Don't use `-s`/`-n` for JIT workloads. Use NVTX or skip filtering and profile all. |

## Step 5: Analyze Results

Read the JSON output. When `--page raw` gives only section-level summaries and you need per-metric breakdown, re-run with `--page details` to get full text output with individual metric values.

Present in this structure:

### 1. Roofline Classification

| Bottleneck | Condition |
|------------|-----------|
| Memory Bound | memory_sol >= compute_sol |
| Compute Bound | compute_sol > memory_sol |
| Underutilized | both < 60% |

Include SOL %, headroom %, tensor core active status.

### 2. Metric Groups

For each of the 10 groups below, show `| Metric | Value | Interpretation |`.
Interpret values using `${CLAUDE_SKILL_DIR}/../gpu-refs/ncu_metrics.md`.

1. **Speed of Light (SOL)** — roofline overview
2. **SM & Compute Utilization** — FP32/tensor/LSU pipe usage
3. **DRAM (HBM)** — bandwidth, bytes read/written
4. **L2 Cache** — hit rate, throughput
5. **L1/TEX Cache** — hit rate, shared memory wavefronts
6. **Global Memory Access** — sectors/request, coalescing, branch divergence
7. **Occupancy & Resources** — limiter (registers/shared_mem/warps)
8. **Launch Configuration** — block/grid size, registers, shared mem, waves/SM
9. **Stall Analysis** — memory_dependency, scoreboard, barrier, branch
10. **Timing** — duration, active time

### 3. Derived Metrics

- **Global Load Sectors/Request**: 1.0 = perfect, >4 = waste, 32 = worst
- **Global Store Sectors/Request**: same scale
- **L2 Hit Rate** (computed): cross-check with reported
- **DRAM Bandwidth (GB/s)**: compare with GPU peak
- **Kernel Duration (ms)**: absolute time

### 4. Code-Level Mapping

Read kernel source. Map findings to specific code using `${CLAUDE_SKILL_DIR}/../gpu-refs/bottleneck_patterns.md`.

For each major finding:
- Quote the relevant code section
- Explain what metrics suggest about that code
- State confidence: **certain** / **likely** / **speculative**

GPU DSL compilers (Triton/Numba/TileLang/nvcc) and JIT paths (load_inline/cpp_extension) transform code through multiple passes, so exact source-to-SASS mapping is imprecise.

### 5. Key Findings

Top 3-5 findings ordered by estimated impact:

1. **What**: the bottleneck
2. **Evidence**: specific metric values
3. **Where**: code location if identifiable
4. **Impact**: High / Medium / Low

**Do NOT suggest fixes unless asked.**

## Follow-Up

After presenting results, offer:
- "Open `.ncu-rep` in Nsight Compute GUI?" (give path)
- "Re-run with NVTX to isolate a specific kernel?"
- "Re-run with `--page details` for per-metric breakdown?"
- "Run `/nsys-profile` for timeline and GPU utilization data?"
- "Dive deeper into any metric group?"

Always report the `.ncu-rep` path for GUI inspection.
