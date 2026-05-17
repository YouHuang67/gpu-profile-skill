<div align="right">
  <b>中文</b> | <a href="README.md">English</a>
</div>

<br>

<h1 align="center">GPU Profile Skill</h1>

<p align="center">
  <b>面向 Claude Code 和 Codex 的 GPU kernel 性能分析工具</b><br>
  <sub>NCU 硬件计数器 &middot; NSYS 时间线分析 &middot; Triton &middot; CUDA &middot; TileLang</sub>
</p>

<p align="center">
  <a href="https://github.com/your-org/gpu-profile-skill/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="MIT License">
  </a>
  <a href="#安装">
    <img src="https://img.shields.io/badge/安装-30_秒-blue?style=flat-square" alt="安装">
  </a>
  <a href="#支持的框架">
    <img src="https://img.shields.io/badge/框架-12-green?style=flat-square" alt="12 种框架">
  </a>
</p>

---

## 概述

一套 Agent Skills 包，让 coding agent 能够调用 NVIDIA Nsight Compute 和 Nsight Systems 对 GPU kernel 进行性能分析。agent 自动检测 kernel 框架（Triton、CUDA、TileLang 等 12 种），选择合适的 profiling 工具，输出结构化分析报告。

agent 负责处理环境检查、sudo 权限检测、CUDA 工具路径发现、kernel 入口识别、profiling 执行、指标解读、代码级瓶颈映射等全流程。

## 安装

<h3>环境要求</h3>

<table>
<tr><td width="160"><b>NVIDIA GPU</b></td><td>计算能力 7.0+</td></tr>
<tr><td><b>CUDA Toolkit</b></td><td>11.8+，需包含 <a href="https://developer.nvidia.com/nsight-compute">Nsight Compute</a> 和 <a href="https://developer.nvidia.com/nsight-systems">Nsight Systems</a></td></tr>
<tr><td><b>Python</b></td><td>3.7+，需安装 <code>torch</code>（CUDA 版本）</td></tr>
<tr><td><b>运行环境</b></td><td>Claude Code 或 Codex（Agent Skills 兼容）</td></tr>
</table>

<h3>Sudo 配置（一次性）</h3>

NCU 读取 GPU 硬件性能计数器需要 root 权限，做一次永久配置即可：

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u
```

然后**重启系统**：

```bash
sudo reboot
```

重启后验证 `cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly`，期望输出 `0`。如果跳过此配置，profiling 脚本每次 NCU 调用时会自动使用 sudo。

<h3>安装</h3>

```bash
git clone https://github.com/your-org/gpu-profile-skill.git
cd gpu-profile-skill
./install.sh check      # 检查 GPU、CUDA、NCU、NSYS、sudo 状态
./install.sh install     # 软链接到 ~/.claude/skills/
```

Codex 用户：`./install.sh install -t codex`。

## 使用方式

安装后直接对 agent 描述你的需求，agent 会根据意图触发对应的分析流程：

- 性能问题：*"这个 kernel 为什么这么慢"*、*"吞吐上限在哪"*、*"看一下瓶颈"*
- 分析请求：*"profile 一下 attention kernel"*、*"跑个 NCU"*、*"检查占用率"*
- 具体指标：*"tensor core 利用率"*、*"内存带宽"*、*"stall 分析"*

agent 输出 roofline 分类、10 组硬件指标、派生指标、带可信度标注的代码级映射、按影响排序的关键发现。

对于 JIT 编译和预编译 kernel（缺少 Python 装饰器的场景），NCU 仍能捕获所有 CUDA launch。agent 在运行时自动发现 kernel 名称，不依赖写死的规则。sandbox 目录下有各类入口模式的参考示例。

## 支持的框架

| 框架 | 检测方式 |
|------|----------|
| **Triton** | `@triton.jit`、`@triton.autotune` |
| **Numba CUDA** | `@cuda.jit` |
| **TileLang** | `@tilelang.jit`、`@T.prim_func` |
| **PyTorch JIT** | `load_inline`、`cpp_extension.load` |
| **CuPy** | `RawKernel`、`RawModule` |
| **PyCUDA** | `SourceModule` |
| **预编译 .so** | `ctypes.CDLL`、`import xxx_cuda` |
| **Triton AOT** | `triton.compile` |
| **PyTorch CUDA** | `import torch` + CUDA |
| **JAX, TensorRT, TensorFlow** | 框架特有调用 |

## 项目结构

```
skills/
├── ncu-profile/        # NCU 硬件计数器分析
│   ├── SKILL.md
│   ├── scripts/        # -> ../../shared/scripts/
│   └── reference/      # -> ../../shared/reference/
├── nsys-profile/       # NSYS 时间线分析
├── profile/            # NCU + NSYS 组合分析
└── shared/             # 源文件，不安装为 skill
    ├── scripts/        # detect_entry, run_ncu, run_nsys
    └── reference/      # 指标解读、瓶颈映射、SQLite schema
```

每个 skill 是一个自包含目录，符合 Agent Skills 标准。脚本和参考文档在 `shared/` 下存放一份，通过 symlink 链接到各 skill 目录。

## License

MIT。详见 [LICENSE](LICENSE)。
