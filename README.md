<div align="right">
  <a href="README_CN.md">中文</a> | <b>English</b>
</div>

<br>

<h1 align="center">GPU Profile Skill</h1>

<p align="center">
  <b>One-command GPU kernel profiling, works with Claude Code and Codex</b><br>
  <sub>NCU hardware counters &middot; NSYS timeline tracing &middot; Triton &middot; CUDA &middot; TileLang</sub>
</p>

<p align="center">
  <a href="https://github.com/your-org/gpu-profile-skill/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="MIT License">
  </a>
  <a href="#install">
    <img src="https://img.shields.io/badge/Install-30s-blue?style=flat-square" alt="Install in 30s">
  </a>
  <a href="#supported-frameworks">
    <img src="https://img.shields.io/badge/Frameworks-Triton%20%7C%20CUDA%20%7C%20TileLang-green?style=flat-square" alt="Frameworks">
  </a>
</p>

---

## What This Does

Three slash commands that wrap NVIDIA Nsight tooling into automated profiling workflows. Compatible with Claude Code and Codex via the Agent Skills standard. No config files, no boilerplate. Point at a Python file that launches GPU kernels and get structured performance analysis back.

<table>
<tr>
  <td width="130"><b><code>/ncu-profile</code></b></td>
  <td>NCU hardware counter deep-dive: roofline, occupancy, stalls, cache, coalescing, tensor core utilization</td>
  <td align="center">May need sudo</td>
</tr>
<tr>
  <td><b><code>/nsys-profile</code></b></td>
  <td>NSYS timeline analysis: GPU utilization, kernel timing, memory transfers, launch overhead, CPU/GPU overlap</td>
  <td align="center">No sudo</td>
</tr>
<tr>
  <td><b><code>/profile</code></b></td>
  <td>Full combo: NSYS first for a fast overview, then NCU for hardware deep-dive, then unified report</td>
  <td align="center">Depends on NCU</td>
</tr>
</table>

**Supported frameworks**: Triton, Numba CUDA, TileLang, PyCUDA, CuPy, raw PyTorch CUDA.

## What You Get

The agent detects your framework, checks the environment, finds the kernel entry point, runs profiling, and presents a structured report:

```
Roofline  >  10 metric groups  >  code-level mapping  >  prioritized findings
```

Each finding maps hardware metrics to specific code locations, with certainty levels. **Data for your optimization decisions, not automated rewrites.**

## Install

<h3>1. Prerequisites</h3>

<table>
<tr><td width="180"><b>NVIDIA GPU</b></td><td>Compute capability 7.0+</td></tr>
<tr><td><b>CUDA Toolkit</b></td><td>11.8+ with <a href="https://developer.nvidia.com/nsight-compute">Nsight Compute</a> and <a href="https://developer.nvidia.com/nsight-systems">Nsight Systems</a></td></tr>
<tr><td><b>Python</b></td><td>3.7+ with <code>torch</code> (CUDA build)</td></tr>
<tr><td><b>Agent runtime</b></td><td>Claude Code or Codex (Agent Skills compatible)</td></tr>
</table>

<h3>2. One-time sudo setup</h3>

NCU reads GPU hardware performance counters, which default to root-only. Fix once (reboot required):

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u && sudo reboot
```

After reboot, run `cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly`; expected output is `0`.

If you skip this step, profiling scripts auto-use `sudo` (you will get password prompts for each NCU run).

<h3>3. Install skills</h3>

```bash
git clone https://github.com/your-org/gpu-profile-skill.git
cd gpu-profile-skill
./install.sh check      # verify GPU, CUDA, NCU, NSYS, sudo status
./install.sh install     # symlink skills into ~/.claude/skills/
```

Uninstall: `./install.sh uninstall`. Check status: `./install.sh status`.

After install, `/ncu-profile`, `/nsys-profile`, `/profile` are available in your agent.

<h3>Codex install</h3>

```bash
./install.sh install -t codex     # -> ./.agents/skills/
./install.sh status -t codex
```

## Usage

```
/profile path/to/kernel.py                           # Full NCU + NSYS
/ncu-profile path/to/kernel.py                       # Hardware counters only
/nsys-profile path/to/kernel.py                      # Timeline only
/ncu-profile path/to/kernel.py -k "sparse_attn"      # Filter to specific kernel
/ncu-profile path/to/test.py --nvtx profile_target   # NVTX: skip autotune warmup
```

The entry detector (`detect_entry.py`) auto-classifies the file as runnable, callable, or kernel-only and generates wrappers when needed.

## Architecture

```
skills/
├── ncu-profile/SKILL.md        # /ncu-profile command
├── nsys-profile/SKILL.md       # /nsys-profile command
├── profile/SKILL.md            # /profile command (NCU + NSYS)
├── gpu-refs/                   # Shared docs: metrics + bottleneck mapping
│   ├── ncu_metrics.md
│   ├── nsys_metrics.md
│   └── bottleneck_patterns.md
└── gpu-scripts/                # Shared Python tools
    ├── detect_entry.py         # Multi-framework AST entry detection
    ├── run_ncu.py              # NCU execution + metric extraction + roofline
    └── run_nsys.py             # NSYS execution + timeline parsing + SQLite
```

Each skill directory contains a single `SKILL.md`. Shared resources (`gpu-refs/`, `gpu-scripts/`) have no `SKILL.md` so the agent ignores them as commands, while keeping them accessible internally via `../gpu-refs/` and `../gpu-scripts/` paths.

## Supported Frameworks

| Framework | Detected by |
|-----------|-------------|
| **Triton** | `@triton.jit`, `@triton.autotune`, `import triton` |
| **Numba CUDA** | `@cuda.jit`, `from numba import cuda` |
| **TileLang** | `@tilelang.jit`, `@T.prim_func`, `import tilelang` |
| **PyCUDA** | `import pycuda`, `SourceModule` |
| **CuPy** | `import cupy` |
| **PyTorch CUDA** | `import torch` + CUDA operations |

Works for `.py` files. If your kernel is `.cu`, provide a Python wrapper that launches it.

## License

MIT. See [LICENSE](LICENSE).
