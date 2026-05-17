<div align="right">
  <a href="README_CN.md">中文</a> | <b>English</b>
</div>

<br>

<h1 align="center">GPU Profile Skill</h1>

<p align="center">
  <b>GPU kernel profiling for Claude Code and Codex</b><br>
  <sub>NCU hardware counters &middot; NSYS timeline tracing &middot; Triton &middot; CUDA &middot; TileLang</sub>
</p>

<p align="center">
  <a href="https://github.com/YouHuang67/gpu-profile-skill/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="MIT License">
  </a>
  <a href="#install">
    <img src="https://img.shields.io/badge/Install-30s-blue?style=flat-square" alt="Install">
  </a>
  <a href="#frameworks">
    <img src="https://img.shields.io/badge/Frameworks-12-green?style=flat-square" alt="12 frameworks">
  </a>
</p>

---

## Overview

An Agent Skills package that gives your coding agent the ability to profile GPU kernels with NVIDIA Nsight Compute and Nsight Systems. The agent detects the kernel framework (Triton, CUDA, TileLang, and 9 others), runs the right profiling tool, and produces a structured analysis report.

The agent handles environment checks, sudo permission detection, CUDA binary discovery, kernel entry point detection, profiling execution, metric interpretation, and code-level bottleneck mapping.

## Install

<h3>Prerequisites</h3>

<table>
<tr><td width="180"><b>NVIDIA GPU</b></td><td>Compute capability 7.0+</td></tr>
<tr><td><b>CUDA Toolkit</b></td><td>11.8+ with <a href="https://developer.nvidia.com/nsight-compute">Nsight Compute</a> and <a href="https://developer.nvidia.com/nsight-systems">Nsight Systems</a></td></tr>
<tr><td><b>Python</b></td><td>3.7+ with <code>torch</code> (CUDA build)</td></tr>
<tr><td><b>Agent runtime</b></td><td>Claude Code or Codex (Agent Skills compatible)</td></tr>
</table>

<h3>Sudo (one-time)</h3>

NCU reads GPU hardware performance counters, which default to root-only. Fix once:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u
```

Then **reboot**:

```bash
sudo reboot
```

After reboot, verify: `cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly`. Expected output: `0`. If this step is skipped, profiling scripts auto-use sudo at each NCU invocation.

<h3>Install</h3>

```bash
git clone https://github.com/YouHuang67/gpu-profile-skill.git
cd gpu-profile-skill
./install.sh check      # verify GPU, CUDA, NCU, NSYS, sudo
./install.sh install     # symlinks to ~/.claude/skills/
```

For Codex: `./install.sh install -t codex`.

## Usage

After install, ask the agent to profile a kernel. The agent invokes the appropriate skill based on your request:

- Performance questions: *"why is this kernel slow"*, *"what limits throughput"*, *"check the bottleneck"*
- Profiling requests: *"profile the attention kernel"*, *"run NCU on this"*, *"check occupancy"*
- Specific metrics: *"tensor core utilization"*, *"memory bandwidth"*, *"stall analysis"*

The agent produces a report covering roofline classification, 10 metric groups, derived metrics, code-level mapping with confidence labels, and prioritized findings.

For JIT-compiled and pre-compiled kernels that lack Python-level decorators, NCU captures all CUDA launches regardless. The agent discovers kernel names at runtime without hardcoded filters. See the sandbox examples for the range of patterns handled.

## Frameworks

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

## Structure

```
skills/
├── ncu-profile/        # NCU hardware counter analysis
│   ├── SKILL.md
│   ├── scripts/        # -> ../../shared/scripts/
│   └── reference/      # -> ../../shared/reference/
├── nsys-profile/       # NSYS timeline analysis
├── profile/            # Combined NCU + NSYS
└── shared/             # Source files, not installed as a skill
    ├── scripts/        # detect_entry, run_ncu, run_nsys
    └── reference/      # metrics, bottleneck patterns, SQLite schema
```

Each skill is a self-contained directory per the Agent Skills standard. Scripts and reference docs live once under `shared/` and are symlinked into each skill directory.

## License

MIT. See [LICENSE](LICENSE).
