<div align="right">
  <b>中文</b> | <a href="README.md">English</a>
</div>

<br>

<h1 align="center">GPU Profile Skill</h1>

<p align="center">
  <b>一键 GPU Kernel 性能分析，兼容 Claude Code 和 Codex</b><br>
  <sub>NCU 硬件计数器 &middot; NSYS 时间线分析 &middot; Triton &middot; CUDA &middot; TileLang</sub>
</p>

<p align="center">
  <a href="https://github.com/your-org/gpu-profile-skill/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="MIT License">
  </a>
  <a href="#安装">
    <img src="https://img.shields.io/badge/安装-30_秒-blue?style=flat-square" alt="30 秒安装">
  </a>
  <a href="#支持的框架">
    <img src="https://img.shields.io/badge/框架-Triton_|_CUDA_|_TileLang-green?style=flat-square" alt="框架支持">
  </a>
</p>

---

## 这是什么

三个 slash command，把 NVIDIA Nsight 工具链封装成自动化的 profiling 流程。遵循 Agent Skills 标准，兼容 Claude Code 和 Codex。不需要写配置文件，直接指定一个运行 GPU kernel 的 Python 文件，就能得到结构化的性能分析报告。

<table>
<tr>
  <td width="130"><b><code>/ncu-profile</code></b></td>
  <td>NCU 硬件计数器深度分析：roofline、占用率、stall、缓存命中率、内存合并访问、Tensor Core 利用率</td>
  <td align="center">可能需要 sudo</td>
</tr>
<tr>
  <td><b><code>/nsys-profile</code></b></td>
  <td>NSYS 时间线分析：GPU 利用率、kernel 耗时、内存传输、启动开销、CPU 与 GPU 任务是否并行</td>
  <td align="center">不需要 sudo</td>
</tr>
<tr>
  <td><b><code>/profile</code></b></td>
  <td>综合分析：先跑 NSYS 快速扫一眼整体情况，再跑 NCU 深入硬件细节，最后给出统一报告</td>
  <td align="center">取决于 NCU</td>
</tr>
</table>

**支持框架**：Triton、Numba CUDA、TileLang、PyCUDA、CuPy、PyTorch 直接写的 CUDA 代码。

## 能产出什么

Agent 自动检测你用的框架、检查环境、找到 kernel 入口、运行 profiling，输出结构化报告：

```
Roofline 分类  >  10 组硬件指标解读  >  指标对应到哪段代码  >  关键发现（按影响排序）
```

每条发现都把硬件指标映射到具体的代码位置，并标注置信度。**提供数据支撑你的优化决策，不会擅自改代码。**

## 安装

<h3>1. 环境要求</h3>

<table>
<tr><td width="160"><b>NVIDIA GPU</b></td><td>计算能力 7.0 及以上</td></tr>
<tr><td><b>CUDA Toolkit</b></td><td>11.8+，需包含 <a href="https://developer.nvidia.com/nsight-compute">Nsight Compute</a> 和 <a href="https://developer.nvidia.com/nsight-systems">Nsight Systems</a></td></tr>
<tr><td><b>Python</b></td><td>3.7+，需安装 <code>torch</code>（CUDA 版本）</td></tr>
<tr><td><b>运行环境</b></td><td>Claude Code 或 Codex（Agent Skills 兼容）</td></tr>
</table>

<h3>2. 一次性 sudo 配置</h3>

NCU 需要读取 GPU 硬件性能计数器，默认只有 root 能访问。做一次永久配置就好（需要重启）：

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u && sudo reboot
```

重启后执行 `cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly`，应该输出 `0`。

如果跳过这一步，profiling 脚本会自动加 `sudo` 执行 NCU（每次运行时弹出密码提示）。

<h3>3. 安装 skill</h3>

```bash
git clone https://github.com/your-org/gpu-profile-skill.git
cd gpu-profile-skill
./install.sh check      # 先检查 GPU、CUDA、NCU、NSYS、sudo 状态
./install.sh install     # 把 skill 软链接到 ~/.claude/skills/
```

卸载：`./install.sh uninstall`，查看状态：`./install.sh status`。

装好之后直接在 agent 中使用 `/ncu-profile`、`/nsys-profile`、`/profile`。

<h3>Codex 安装</h3>

```bash
./install.sh install -t codex     # -> ./.agents/skills/
./install.sh status -t codex
```

## 使用方式

```
/profile path/to/kernel.py                           # 全量：NCU + NSYS
/ncu-profile path/to/kernel.py                       # 只看硬件计数器
/nsys-profile path/to/kernel.py                      # 只看时间线
/ncu-profile path/to/kernel.py -k "sparse_attn"      # 只分析名字匹配的 kernel
/ncu-profile path/to/test.py --nvtx profile_target   # 用 NVTX 标记，跳过 autotune 预热
```

入口检测脚本（`detect_entry.py`）自动识别文件类型：可直接运行、有启动函数需要 wrapper、纯 kernel 定义需要你指定调用方式。需要 wrapper 的时候会自动生成。

## 项目结构

```
skills/
├── ncu-profile/SKILL.md        # /ncu-profile 命令
├── nsys-profile/SKILL.md       # /nsys-profile 命令
├── profile/SKILL.md            # /profile 命令（NCU + NSYS 组合）
├── gpu-refs/                   # 共享参考文档：指标解读 + 瓶颈映射
│   ├── ncu_metrics.md
│   ├── nsys_metrics.md
│   └── bottleneck_patterns.md
└── gpu-scripts/                # 共享 Python 脚本
    ├── detect_entry.py         # 多框架 AST 入口检测
    ├── run_ncu.py              # NCU 执行 + 指标提取 + roofline 分类
    └── run_nsys.py             # NSYS 执行 + 时间线解析 + SQLite 分析
```

每个 skill 目录只放一个 `SKILL.md`。`gpu-refs/` 和 `gpu-scripts/` 没有 `SKILL.md`，agent 不会把它们当成命令列出来，但 skill 内部可以通过 `../gpu-refs/` 和 `../gpu-scripts/` 访问。

## 支持的框架

| 框架 | 检测方式 |
|------|----------|
| **Triton** | `@triton.jit`、`@triton.autotune`、`import triton` |
| **Numba CUDA** | `@cuda.jit`、`from numba import cuda` |
| **TileLang** | `@tilelang.jit`、`@T.prim_func`、`import tilelang` |
| **PyCUDA** | `import pycuda`、`SourceModule` |
| **CuPy** | `import cupy` |
| **PyTorch CUDA** | `import torch` + CUDA 操作 |

目前只支持 `.py` 文件。如果你的 kernel 是 `.cu` 写的，提供一个 Python wrapper 来调用即可。

## License

MIT。详见 [LICENSE](LICENSE)。
