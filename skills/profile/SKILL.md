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

# Profile — Combined NCU + NSYS

Orchestrates NCU (hardware counters) + NSYS (timeline) for complete performance picture.

**Rule**: Only profile when user explicitly asks.

## Common Blockers
### Sudo for NCU
```bash
cat /proc/driver/nvidia/params 2>/dev/null | grep RmProfilingAdminOnly
```
If `RmProfilingAdminOnly: 1`: auto-sudo. Permanent: `echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf && sudo update-initramfs -u && sudo reboot`

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

## Step 1: Environment & Entry
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
python scripts/detect_entry.py "$ARGUMENTS" --json
```

## Step 2: Ask What to Run
1. Full (NCU + NSYS)  2. NCU only  3. NSYS only

## Step 3: NSYS First
```bash
python scripts/run_nsys.py <script> -o $PROFILE_DIR/nsys --python "$(which python)"
```

## Step 4: NCU
```bash
python scripts/run_ncu.py <script> -o $PROFILE_DIR/ncu --python "$(which python)"
```

## Step 5: Unified Report
1. Executive Summary
2. Timeline Overview (NSYS)
3. Hardware Bottleneck (NCU)
4. Code-Level Mapping (`reference/bottleneck_patterns.md`, `reference/ncu_metrics.md`, `reference/nsys_metrics.md`)
5. Key Findings (prioritized, each: What + Evidence + Where + Impact)

Do NOT suggest fixes unless asked.

## Follow-Up
- "Open .ncu-rep / .nsys-rep in GUI?" (provide paths)
- "Run `/ncu-profile` for deeper hardware counter analysis?"
- "Run `/nsys-profile` for focused timeline analysis?"
- "Re-run with different parameters?"

Always report .ncu-rep and .nsys-rep paths. Temp files in $PROFILE_DIR -- remind user.

Always report .ncu-rep and .nsys-rep paths. Remind user about temp files.
