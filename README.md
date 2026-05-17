<div align="right">
  <a href="README_CN.md">中文</a> | <b>English</b>
</div>

<br>

<h1 align="center">GPU Profile Skill</h1>

<p align="center">
  <b>Install, ask, get a report.</b><br>
  <sub>NCU hardware counters &middot; NSYS timeline tracing &middot; Triton &middot; CUDA &middot; TileLang</sub>
</p>

<p align="center">
  <a href="https://github.com/your-org/gpu-profile-skill/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="MIT License">
  </a>
  <a href="#install">
    <img src="https://img.shields.io/badge/Install-30s-blue?style=flat-square" alt="Install">
  </a>
  <a href="#supported-frameworks">
    <img src="https://img.shields.io/badge/Frameworks-12-green?style=flat-square" alt="12 frameworks">
  </a>
</p>

---

## How It Works

You say something like "this kernel is slow, show me the bottleneck" or "profile the attention kernel". The agent detects your framework, runs NCU and NSYS, and presents a structured report: roofline classification, 10 groups of hardware metrics, code-level bottleneck mapping, and prioritized findings.

<table>
<tr>
  <td width="160"><b>Ask about bottlenecks</b></td>
  <td>"why is this kernel slow", "what limits throughput", "is this memory bound"</td>
</tr>
<tr>
  <td><b>Ask for profiling</b></td>
  <td>"profile the attention kernel", "run NCU on this", "check occupancy"</td>
</tr>
<tr>
  <td><b>Ask about specific metrics</b></td>
  <td>"tensor core utilization", "memory bandwidth analysis", "stall analysis"</td>
</tr>
</table>

The agent knows when NCU needs sudo, when to use NVTX for autotune, how to find kernel names in JIT-compiled code, and how to map hardware metrics back to source code.

## What You Get

A structured report from a single prompt:

```
Roofline: memory-bound, 72% Memory SOL, 28% headroom

10 metric groups: SOL, compute, DRAM, L1/L2 cache, coalescing, occupancy,
  launch config, stalls, timing

Code-level mapping: each finding linked to specific source lines with
  confidence labels

Prioritized findings: top bottlenecks ordered by impact, with concrete
  metric values as evidence
```

## Install

<h3>1. Prerequisites</h3>

<table>
<tr><td width="180"><b>NVIDIA GPU</b></td><td>Compute capability 7.0+</td></tr>
<tr><td><b>CUDA Toolkit</b></td><td>11.8+ with <a href="https://developer.nvidia.com/nsight-compute">Nsight Compute</a> and <a href="https://developer.nvidia.com/nsight-systems">Nsight Systems</a></td></tr>
<tr><td><b>Python</b></td><td>3.7+ with <code>torch</code> (CUDA build)</td></tr>
<tr><td><b>Agent runtime</b></td><td>Claude Code or Codex (Agent Skills compatible)</td></tr>
</table>

<h3>2. Sudo (one-time)</h3>

NCU reads hardware performance counters which default to root-only.

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u && sudo reboot
```

Verify: `cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly` should output `0`. If you skip this, profiling scripts auto-use sudo.

<h3>3. Install</h3>

```bash
git clone https://github.com/your-org/gpu-profile-skill.git
cd gpu-profile-skill
./install.sh check      # verify GPU, CUDA, NCU, NSYS, sudo
./install.sh install     # symlinks to ~/.claude/skills/
```

For Codex: `./install.sh install -t codex` (installs to `.agents/skills/`).

## Supported Frameworks

| Framework | Detected by |
|-----------|-------------|
| **Triton** | `@triton.jit`, `@triton.autotune` |
| **Numba CUDA** | `@cuda.jit` |
| **TileLang** | `@tilelang.jit`, `@T.prim_func` |
| **PyTorch JIT** | `load_inline`, `cpp_extension.load` |
| **CuPy** | `RawKernel`, `RawModule` |
| **PyCUDA** | `SourceModule` |
| **Pre-compiled .so** | `ctypes.CDLL`, `import xxx_cuda` |
| **Triton AOT** | `triton.compile` |
| **PyTorch CUDA** | `import torch` + CUDA |
| **JAX, TensorRT, TensorFlow** | framework-specific calls |

For JIT and pre-compiled kernels, NCU captures all CUDA launches regardless. The agent handles kernel name discovery without hardcoded filters.

## Architecture

```
skills/
├── ncu-profile/        # hardware counter analysis
│   ├── SKILL.md
│   ├── scripts/        # -> ../../shared/scripts/
│   └── reference/      # -> ../../shared/reference/
├── nsys-profile/       # timeline analysis
├── profile/            # combined NCU + NSYS
└── shared/             # single source of truth (not installed)
    ├── scripts/        # detect_entry, run_ncu, run_nsys
    └── reference/      # metrics, bottleneck patterns, SQLite schema
```

Each skill is self-contained per the Agent Skills standard.

## License

MIT. See [LICENSE](LICENSE).
