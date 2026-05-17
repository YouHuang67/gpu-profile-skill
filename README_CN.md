<div align="right">
  <b>中文</b> | <a href="README.md">English</a>
</div>

<br>

<h1 align="center">GPU Profile Skill</h1>

<p align="center">
  <b>安装，提问，拿到报告。</b><br>
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

## 怎么用

对 agent 说 "这个 kernel 好慢，看看瓶颈在哪" 或者 "帮我 profile 一下 attention kernel"。agent 自动检测框架，跑 NCU 和 NSYS，给出结构化报告：roofline 分类、10 组硬件指标、代码级瓶颈映射、按影响排序的关键发现。

<table>
<tr>
  <td width="160"><b>问瓶颈</b></td>
  <td>"为什么这个 kernel 这么慢"、"吞吐上限在哪"、"是不是 memory bound"</td>
</tr>
<tr>
  <td><b>要 profile</b></td>
  <td>"profile 一下 attention kernel"、"跑个 NCU"、"检查占用率"</td>
</tr>
<tr>
  <td><b>问具体指标</b></td>
  <td>"tensor core 利用率多少"、"内存带宽分析"、"stall 分析"</td>
</tr>
</table>

agent 知道 NCU 什么时候需要 sudo，autotune 怎么用 NVTX 跳过预热，JIT kernel 名怎么找，怎么把硬件指标映射回源代码。

## 能拿到什么

一次提问，一份结构化报告：

```
Roofline: memory-bound, Memory SOL 72%, 还有 28% 空间

10 组指标：SOL、计算、DRAM、L1/L2 缓存、合并访问、占用率、
  launch 配置、stall、耗时

代码级映射：每条发现关联到具体代码行，标注可信度

关键发现：按影响排序的主要瓶颈，每条带具体指标数值作为证据
```

## 安装

<h3>1. 环境要求</h3>

<table>
<tr><td width="160"><b>NVIDIA GPU</b></td><td>计算能力 7.0+</td></tr>
<tr><td><b>CUDA Toolkit</b></td><td>11.8+，需包含 <a href="https://developer.nvidia.com/nsight-compute">Nsight Compute</a> 和 <a href="https://developer.nvidia.com/nsight-systems">Nsight Systems</a></td></tr>
<tr><td><b>Python</b></td><td>3.7+，需安装 <code>torch</code>（CUDA 版本）</td></tr>
<tr><td><b>运行环境</b></td><td>Claude Code 或 Codex（Agent Skills 兼容）</td></tr>
</table>

<h3>2. Sudo（一次性配置）</h3>

NCU 需要读取 GPU 硬件性能计数器，默认只有 root 能访问。

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u && sudo reboot
```

重启后验证 `cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly`，应该输出 `0`。跳过的话 profiling 脚本会自动加 sudo。

<h3>3. 安装</h3>

```bash
git clone https://github.com/your-org/gpu-profile-skill.git
cd gpu-profile-skill
./install.sh check      # 检查 GPU、CUDA、NCU、NSYS、sudo
./install.sh install     # 软链接到 ~/.claude/skills/
```

Codex 用户：`./install.sh install -t codex`（安装到 `.agents/skills/`）。

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
| **JAX, TensorRT, TensorFlow** | 各自框架特有调用 |

JIT 和预编译 kernel 即使没有 Python 装饰器，NCU 也能捕获全部 CUDA launch。agent 会处理 kernel 名发现，不用写死规则。

## 项目结构

```
skills/
├── ncu-profile/        # 硬件计数器分析
│   ├── SKILL.md
│   ├── scripts/        # -> ../../shared/scripts/
│   └── reference/      # -> ../../shared/reference/
├── nsys-profile/       # 时间线分析
├── profile/            # NCU + NSYS 组合
└── shared/             # 源文件（不安装为 skill）
    ├── scripts/        # detect_entry, run_ncu, run_nsys
    └── reference/      # 指标解读、瓶颈映射、SQLite schema
```

每个 skill 自包含，符合 Agent Skills 标准。

## License

MIT。详见 [LICENSE](LICENSE)。
