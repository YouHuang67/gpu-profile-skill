---
name: profile
description: "/profile: combined NCU + NSYS comprehensive GPU kernel profiling. Triggers: full profile, complete analysis, performance regression, both ncu and nsys, profile this."
triggers:
  - /profile
  - full profile
  - complete profile
  - full profiling
  - comprehensive analysis
  - profile this
  - both ncu and nsys
---

# Profile — Comprehensive GPU Kernel Performance Analysis

Orchestrates NCU (hardware counters) + NSYS (timeline) for a complete performance picture.

**Rule**: Only run profiling when the user explicitly asks. Do NOT suggest it unprompted.

## Common Blockers

### Sudo for NCU
```bash
cat /proc/driver/nvidia/params 2>/dev/null | grep RmProfilingAdminOnly
```
If `RmProfilingAdminOnly: 1` or unreadable → NCU auto-uses sudo.  
Permanent fix: `echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf && sudo update-initramfs -u && sudo reboot`

### NCU/NSYS Binaries
```bash
python -c "
import shutil, os
for tool in ['ncu', 'nsys']:
    p = shutil.which(tool)
    if p: print(f'{tool} (PATH): {p}')
    for v in ['12.8','12.6','12.4','12.2','12.0','11.8']:
        c = f'/usr/local/cuda-{v}/bin/{tool}'
        if os.path.isfile(c): print(f'{tool}: {c}')
"
```

## Step 1: Environment & Entry Detection

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
python ${CLAUDE_SKILL_DIR}/../gpu-scripts/detect_entry.py "$ARGUMENTS" --json
```

If wrapper needed, create shared temp dir:
```bash
PROFILE_DIR="/tmp/gpu_profile_$(date +%s)" && mkdir -p "$PROFILE_DIR"
```

## Step 2: Ask What to Run

1. **Full analysis** (NCU + NSYS) — recommended for first-time profiling
2. **NCU only** — hardware counters (may need sudo)
3. **NSYS only** — timeline (faster, no sudo)

## Step 3: Run NSYS First (if selected)

```bash
python ${CLAUDE_SKILL_DIR}/../gpu-scripts/run_nsys.py \
    <script> -o $PROFILE_DIR/nsys \
    --python "$(which python)"
```

Present brief summary (GPU utilization, top kernels, obvious issues).  
If NSYS fails, report error but continue to NCU.

## Step 4: Run NCU (if selected)

```bash
python ${CLAUDE_SKILL_DIR}/../gpu-scripts/run_ncu.py \
    <script> -o $PROFILE_DIR/ncu \
    --python "$(which python)"
```

With NVTX: `--nvtx-include "profile_target"`  
With explicit path: `--ncu /usr/local/cuda-12.4/bin/ncu`

If NCU fails and NSYS succeeded, still present NSYS results.

## Step 5: Unified Analysis Report

### 1. Executive Summary

One paragraph: kernel type, roofline classification, GPU utilization, top 1-2 bottlenecks.  
If only one tool succeeded, note the analysis is partial.

### 2. Timeline Overview (NSYS)

GPU utilization %, kernel time distribution, memory transfer overhead, launch overhead.  
Skip if NSYS not run or failed.

### 3. Hardware Bottleneck Analysis (NCU)

Roofline with SOL values, top stall contributors, occupancy, cache effectiveness, TC utilization.  
Skip if NCU not run or failed.

### 4. Code-Level Mapping

For each major finding, map to source code:
- Quote the relevant code section
- Explain what metrics suggest
- State confidence: **certain** / **likely** / **speculative**

References:
- `${CLAUDE_SKILL_DIR}/../gpu-refs/bottleneck_patterns.md`
- `${CLAUDE_SKILL_DIR}/../gpu-refs/ncu_metrics.md`
- `${CLAUDE_SKILL_DIR}/../gpu-refs/nsys_metrics.md`

### 5. Key Findings (Prioritized)

Numbered by estimated impact. Each:
1. **What**: the bottleneck/issue
2. **Evidence**: specific metric values
3. **Where**: code location if identifiable
4. **Impact**: High / Medium / Low

Do NOT include fix recommendations unless asked.

## Follow-Up

- "Which finding to investigate further?"
- "Re-run with different parameters?"
- "See raw metric data for a specific group?"
- "Open .ncu-rep / .nsys-rep in GUI?" (provide paths)

Always report `.ncu-rep` and `.nsys-rep` paths.  
Temp files in `$PROFILE_DIR` — remind user at end.
