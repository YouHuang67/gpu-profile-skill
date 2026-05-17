"""Execute NCU profiling on a GPU kernel and parse results.

Runs Nsight Compute, saves .ncu-rep, then extracts CSV via ncu --import.
Computes roofline classification, derived metrics, and outputs structured
JSON for Claude to interpret.

Two-phase approach:
  1. ncu --export → .ncu-rep (binary, replayable, GUI-openable)
  2. ncu --import --csv → parsed metrics
"""

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

NCU_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    "lts__t_sectors_lookup_hit.sum",
    "lts__t_sectors_lookup_miss.sum",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors.sum",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sectors_lookup_hit.sum",
    "l1tex__t_sectors_lookup_miss.sum",
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "smsp__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "smsp__warps_active.avg.per_cycle_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__occupancy_per_block_size",
    "launch__occupancy_per_register_count",
    "launch__occupancy_per_shared_mem_size",
    "launch__block_size",
    "launch__grid_size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_dynamic",
    "launch__shared_mem_per_block_static",
    "launch__waves_per_multiprocessor",
    "gpu__time_duration.sum",
    "gpu__time_active.sum",
    "sm__cycles_elapsed.avg",
    "sm__cycles_active.avg",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
]

METRIC_SECTIONS = {
    "speed_of_light": {
        "label": "Speed of Light (SOL)",
        "metrics": [
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
        ],
    },
    "sm_compute": {
        "label": "SM & Compute Utilization",
        "metrics": [
            "sm__cycles_active.avg",
            "sm__cycles_elapsed.avg",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "sm__warps_active.avg.per_cycle_active",
            "sm__inst_executed.sum",
            "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
            "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
        ],
    },
    "dram_hbm": {
        "label": "DRAM (HBM)",
        "metrics": [
            "dram__bytes.sum",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__bytes.sum.per_second",
            "dram__sectors_read.sum",
            "dram__sectors_write.sum",
        ],
    },
    "l2_cache": {
        "label": "L2 Cache",
        "metrics": [
            "lts__t_sectors_lookup_hit.sum",
            "lts__t_sectors_lookup_miss.sum",
            "lts__t_sector_hit_rate.pct",
            "lts__t_sectors.sum",
            "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        ],
    },
    "l1_cache": {
        "label": "L1/TEX Cache",
        "metrics": [
            "l1tex__t_sectors_lookup_hit.sum",
            "l1tex__t_sectors_lookup_miss.sum",
            "l1tex__t_sector_hit_rate.pct",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
        ],
    },
    "global_memory_access": {
        "label": "Global Memory Access",
        "metrics": [
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
            "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
            "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
            "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
            "smsp__sass_average_branch_targets_threads_uniform.pct",
        ],
    },
    "occupancy_resources": {
        "label": "Occupancy & Resources",
        "metrics": [
            "launch__occupancy_limit_blocks",
            "launch__occupancy_limit_registers",
            "launch__occupancy_limit_shared_mem",
            "launch__occupancy_limit_warps",
            "launch__occupancy_per_block_size",
            "launch__occupancy_per_register_count",
            "launch__occupancy_per_shared_mem_size",
            "smsp__warps_active.avg.pct_of_peak_sustained_active",
        ],
    },
    "launch_config": {
        "label": "Launch Configuration",
        "metrics": [
            "launch__block_size",
            "launch__grid_size",
            "launch__registers_per_thread",
            "launch__shared_mem_per_block_dynamic",
            "launch__shared_mem_per_block_static",
            "launch__waves_per_multiprocessor",
        ],
    },
    "stall_analysis": {
        "label": "Stall Analysis",
        "metrics": [
            "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
            "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
            "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
        ],
    },
    "timing": {
        "label": "Timing",
        "metrics": [
            "gpu__time_duration.sum",
            "gpu__time_active.sum",
        ],
    },
}


def find_ncu(ncu_bin: str | None = None) -> str:
    """Locate the ncu binary.

    Search order:
    1. Explicit path (--ncu argument)
    2. PATH lookup
    3. $CUDA_HOME/bin
    4. Common CUDA installation directories
    """
    if ncu_bin and ncu_bin != "ncu":
        if os.path.isfile(ncu_bin) and os.access(ncu_bin, os.X_OK):
            return ncu_bin
        found = shutil.which(ncu_bin)
        if found:
            return found
        raise RuntimeError(f"ncu not found at: {ncu_bin}")

    found = shutil.which("ncu")
    if found:
        return found

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    search_dirs = []
    if cuda_home:
        search_dirs.append(os.path.join(cuda_home, "bin"))

    search_dirs.extend([
        "/usr/local/cuda/bin",
    ])
    for ver in ["12.8", "12.7", "12.6", "12.5", "12.4", "12.3",
                "12.2", "12.1", "12.0", "11.8"]:
        search_dirs.append(f"/usr/local/cuda-{ver}/bin")

    for d in search_dirs:
        candidate = os.path.join(d, "ncu")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise RuntimeError(
        "ncu not found. Searched:\n"
        "  - PATH\n"
        f"  - $CUDA_HOME={cuda_home or '(not set)'}\n"
        f"  - /usr/local/cuda*/bin\n"
        "Fix: install Nsight Compute, or pass --ncu /path/to/ncu"
    )


def check_profiling_permission() -> dict:
    """Check if NCU can run without sudo.

    Returns dict with:
        needs_sudo: bool
        message: str - human-readable explanation
        fix_command: str - command to fix (if needs_sudo)
    """
    fix_cmd = (
        "echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | "
        "sudo tee /etc/modprobe.d/nvidia-profiling.conf && sudo reboot"
    )
    result = {"needs_sudo": True, "message": "", "fix_command": fix_cmd}

    param_path = "/proc/driver/nvidia/params"
    if os.path.isfile(param_path):
        try:
            text = Path(param_path).read_text()
            if "RmProfilingAdminOnly: 0" in text:
                result["needs_sudo"] = False
                result["message"] = (
                    "Profiling OK (RmProfilingAdminOnly=0)"
                )
                result["fix_command"] = ""
                return result
        except PermissionError:
            pass

    modprobe_conf = "/etc/modprobe.d/nvidia-profiling.conf"
    if os.path.isfile(modprobe_conf):
        try:
            text = Path(modprobe_conf).read_text()
            if "NVreg_RestrictProfilingToAdminUsers=0" in text:
                result["needs_sudo"] = False
                result["message"] = (
                    "Profiling configured via modprobe "
                    "(reboot required if not yet active)"
                )
                result["fix_command"] = ""
                return result
        except PermissionError:
            pass

    result["message"] = (
        "NCU requires elevated privileges to access GPU performance "
        "counters. Will attempt with sudo.\n"
        "To fix permanently (one-time, requires reboot):\n"
        f"  {fix_cmd}"
    )
    return result


def build_ncu_command(
    script_path: str,
    output_path: str,
    python_bin: str = "python",
    launch_skip: int = 0,
    launch_count: int = 0,
    use_sudo: bool = False,
    ncu_bin: str = "ncu",
    kernel_filter: str = "",
    nvtx_include: str = "",
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the NCU profiling command (phase 1: capture to .ncu-rep)."""
    metrics_str = ",".join(NCU_METRICS)
    cmd = []
    if use_sudo:
        cmd.append("sudo")
    cmd.extend([
        ncu_bin,
        "--export", output_path,
        "--force-overwrite",
        "--target-processes", "all",
        f"--metrics", metrics_str,
    ])

    if kernel_filter:
        cmd.extend(["--kernel-name", kernel_filter])

    if nvtx_include:
        cmd.append("--nvtx")
        cmd.extend(["--nvtx-include", f"{nvtx_include}/"])

    if launch_skip > 0:
        cmd.extend(["--launch-skip", str(launch_skip)])

    if launch_count > 0:
        cmd.extend(["--launch-count", str(launch_count)])

    if extra_args:
        cmd.extend(extra_args)

    cmd.extend([python_bin, script_path])
    return cmd


def build_import_command(
    ncu_rep_path: str, ncu_bin: str = "ncu"
) -> list[str]:
    """Build the ncu --import command (phase 2: extract CSV)."""
    return [ncu_bin, "--import", ncu_rep_path, "--csv", "--page", "raw"]


def run_ncu(
    script_path: str,
    output_dir: str,
    python_bin: str = "python",
    launch_skip: int = 0,
    launch_count: int = 0,
    use_sudo: bool = False,
    ncu_bin: str = "ncu",
    timeout: int = 300,
    kernel_filter: str = "",
    nvtx_include: str = "",
    selection: str = "last",
    kernel_name_filter: str = "",
) -> dict:
    """Run NCU profiling in two phases and return parsed metrics."""
    ncu_bin = find_ncu(ncu_bin)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ncu_rep_path = str(output_dir / "profile.ncu-rep")

    perm = check_profiling_permission()
    if perm["needs_sudo"] and not use_sudo:
        use_sudo = True

    cmd = build_ncu_command(
        script_path, ncu_rep_path, python_bin,
        launch_skip, launch_count, use_sudo, ncu_bin,
        kernel_filter, nvtx_include,
    )

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(script_path).parent),
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"NCU timed out after {timeout}s",
            "command": " ".join(cmd),
        }

    if not Path(ncu_rep_path).exists():
        hints = []
        if proc.returncode == 137:
            hints.append(
                "Process killed (exit 137). Likely causes:\n"
                "  - Out of memory: reduce input data size\n"
                "  - Killed by system: check dmesg for OOM killer"
            )
        stderr_lower = (proc.stderr or "").lower()
        if "permission" in stderr_lower or "denied" in stderr_lower:
            hints.append(perm["message"])
            if perm["fix_command"]:
                hints.append(f"Fix: {perm['fix_command']}")
        if "no kernel" in stderr_lower:
            hints.append(
                "No CUDA kernels were launched. Verify the script "
                "actually runs a GPU kernel."
            )
        if "module" in stderr_lower and "not found" in stderr_lower:
            hints.append(
                "Python module not found. Check that all dependencies "
                "are installed in the conda env and PYTHONPATH is correct."
            )
        if not hints:
            hints.append(
                "NCU failed without a recognized error pattern. "
                "Check stderr output below for details."
            )
        return {
            "success": False,
            "error": "NCU did not produce .ncu-rep file",
            "returncode": proc.returncode,
            "stderr": proc.stderr[-3000:] if proc.stderr else "",
            "hints": hints,
            "command": " ".join(cmd),
        }

    import_cmd = build_import_command(ncu_rep_path, ncu_bin)
    try:
        import_proc = subprocess.run(
            import_cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "ncu --import timed out",
            "ncu_rep_path": ncu_rep_path,
        }

    if import_proc.returncode != 0:
        return {
            "success": False,
            "error": f"ncu --import failed (exit {import_proc.returncode})",
            "ncu_rep_path": ncu_rep_path,
            "stderr": import_proc.stderr[-2000:] if import_proc.stderr else "",
        }

    csv_text = import_proc.stdout
    all_kernels = parse_ncu_csv(csv_text)

    if not all_kernels:
        return {
            "success": False,
            "error": "No kernel data found in NCU output",
            "ncu_rep_path": ncu_rep_path,
        }

    selected = select_kernel(
        all_kernels, kernel_name_filter, selection
    )
    derived = compute_derived_metrics(selected)
    roofline = classify_roofline(selected)
    grouped = group_metrics(selected)

    return {
        "success": True,
        "ncu_rep_path": ncu_rep_path,
        "permission": perm,
        "kernel_name": selected.get("__kernel_name__", "unknown"),
        "all_kernel_names": [
            k.get("__kernel_name__", "unknown") for k in all_kernels
        ],
        "metrics": {
            k: v for k, v in selected.items() if not k.startswith("__")
        },
        "derived": derived,
        "roofline": roofline,
        "grouped_metrics": grouped,
        "command": " ".join(cmd),
    }


def parse_ncu_csv(csv_text: str) -> list[dict[str, str]]:
    """Parse ncu --csv output, auto-detecting format.

    Handles both long format (one row per metric) and wide format (one row per kernel).
    """
    lines = [
        line for line in csv_text.strip().split("\n")
        if line.strip() and not line.startswith("==")
    ]
    if not lines:
        return []

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    rows = list(reader)
    if not rows:
        return []

    if "Metric Name" in rows[0]:
        kernels: dict[str, dict] = defaultdict(dict)
        for row in rows:
            kname = row.get("Kernel Name", "unknown")
            mname = row.get("Metric Name", "")
            mvalue = row.get("Metric Value", "")
            if mname:
                kernels[kname][mname] = mvalue
                kernels[kname]["__kernel_name__"] = kname
        return list(kernels.values())
    else:
        result = []
        for row in rows:
            d = dict(row)
            kname = d.pop("Kernel Name", d.pop("kernel_name", "unknown"))
            d["__kernel_name__"] = kname
            result.append(d)
        return result


def select_kernel(
    kernels: list[dict],
    name_filter: str = "",
    selection: str = "last",
) -> dict:
    """Select a single kernel from parsed results."""
    if name_filter:
        filtered = [
            k for k in kernels
            if name_filter in k.get("__kernel_name__", "")
        ]
        if filtered:
            kernels = filtered

    if selection == "first":
        return kernels[0]
    elif selection == "last":
        return kernels[-1]
    else:
        return kernels[-1]


def _to_float(val) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = str(val).replace(",", "").replace("%", "").strip()
    for suffix in ["bytes", "byte", "sectors", "sector", "ns"]:
        cleaned = cleaned.replace(suffix, "").strip()
    if not cleaned or cleaned.lower() in ("n/a", "nan"):
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def compute_derived_metrics(metrics: dict) -> dict:
    """Compute derived metrics from raw NCU data."""
    derived = {}

    ld_sectors = _to_float(
        metrics.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", 0)
    )
    ld_requests = _to_float(
        metrics.get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", 0)
    )
    st_sectors = _to_float(
        metrics.get("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", 0)
    )
    st_requests = _to_float(
        metrics.get("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", 0)
    )

    if ld_requests > 0:
        derived["global_load_sectors_per_request"] = round(
            ld_sectors / ld_requests, 2
        )
    if st_requests > 0:
        derived["global_store_sectors_per_request"] = round(
            st_sectors / st_requests, 2
        )

    l2_hit = _to_float(metrics.get("lts__t_sectors_lookup_hit.sum", 0))
    l2_miss = _to_float(metrics.get("lts__t_sectors_lookup_miss.sum", 0))
    if l2_hit + l2_miss > 0:
        derived["l2_hit_rate_computed_pct"] = round(
            100.0 * l2_hit / (l2_hit + l2_miss), 1
        )

    dram_bytes = _to_float(metrics.get("dram__bytes.sum", 0))
    duration_ns = _to_float(metrics.get("gpu__time_duration.sum", 0))
    if duration_ns > 0:
        derived["dram_bandwidth_gbps"] = round(dram_bytes / duration_ns, 1)
        derived["kernel_duration_ms"] = round(duration_ns / 1e6, 4)

    return derived


def classify_roofline(
    metrics: dict,
    underutilized_threshold: float = 60.0,
    tensor_core_threshold: float = 5.0,
) -> dict:
    """Classify bottleneck type from SOL metrics."""
    compute_sol = _to_float(
        metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)
    )
    memory_sol = _to_float(
        metrics.get(
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", 0
        )
    )

    efficiency = max(compute_sol, memory_sol)

    tc_activity = _to_float(
        metrics.get(
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", 0
        )
    )
    uses_tensor_cores = tc_activity > tensor_core_threshold

    if (compute_sol < underutilized_threshold
            and memory_sol < underutilized_threshold):
        bottleneck = "underutilized"
    elif memory_sol >= compute_sol:
        bottleneck = "memory"
    else:
        bottleneck = "compute"

    tc_note = (
        " (tensor cores active)" if uses_tensor_cores
        else " (no tensor core usage)"
    )
    if bottleneck == "underutilized":
        desc = (
            f"Both compute ({compute_sol:.1f}%) and memory "
            f"({memory_sol:.1f}%) SOL are below 60%. The kernel is "
            f"underutilized — likely limited by occupancy, stalls, "
            f"or launch configuration{tc_note}."
        )
    elif bottleneck == "memory":
        desc = (
            f"Memory SOL ({memory_sol:.1f}%) >= Compute SOL "
            f"({compute_sol:.1f}%). The kernel is memory bandwidth "
            f"bound{tc_note}."
        )
    else:
        desc = (
            f"Compute SOL ({compute_sol:.1f}%) > Memory SOL "
            f"({memory_sol:.1f}%). The kernel is compute "
            f"bound{tc_note}."
        )

    return {
        "bottleneck": bottleneck,
        "compute_sol_pct": round(compute_sol, 2),
        "memory_sol_pct": round(memory_sol, 2),
        "efficiency_pct": round(efficiency, 2),
        "headroom_pct": round(100 - efficiency, 2),
        "uses_tensor_cores": uses_tensor_cores,
        "description": desc,
    }


def group_metrics(metrics: dict) -> dict:
    """Group metrics into sections for structured display."""
    grouped = {}
    for section_key, section in METRIC_SECTIONS.items():
        section_metrics = {}
        for m in section["metrics"]:
            val = metrics.get(m)
            if val is not None:
                section_metrics[m] = val
        if section_metrics:
            grouped[section_key] = {
                "label": section["label"],
                "metrics": section_metrics,
            }
    return grouped


def main():
    parser = argparse.ArgumentParser(
        description="Run NCU profiling on a GPU kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Basic profiling
  python run_ncu.py test_kernel.py

  # Profile with NVTX (skip autotune, capture target kernel)
  python run_ncu.py test_kernel.py --nvtx-include profile_target

  # Profile specific kernel, skip warmup
  python run_ncu.py test_kernel.py -k "sparse_attention" -s 5 -n 1

  # Compare: filter different kernel from same profile
  python run_ncu.py test_kernel.py --kernel-name-filter "window_rearrange"
""",
    )
    parser.add_argument("script", help="Python script to profile")
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory (default: auto temp dir)",
    )
    parser.add_argument("--python", default="python", help="Python binary")
    parser.add_argument("-s", "--launch-skip", type=int, default=0)
    parser.add_argument("-n", "--launch-count", type=int, default=0)
    parser.add_argument(
        "--sudo", action="store_true",
        help="Force sudo (auto-detected if needed)",
    )
    parser.add_argument("--ncu", default="ncu", help="NCU binary path")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "-k", "--kernel-filter", default="",
        help="Regex to filter kernel names during profiling",
    )
    parser.add_argument(
        "--nvtx-include", default="",
        help="NVTX range name to capture (enables NVTX mode)",
    )
    parser.add_argument(
        "--selection", default="last",
        choices=["first", "last"],
        help="Which kernel invocation to select from results",
    )
    parser.add_argument(
        "--kernel-name-filter", default="",
        help="Substring filter on kernel name in results",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = tempfile.mkdtemp(prefix="gpu_profile_ncu_")

    result = run_ncu(
        script_path=args.script,
        output_dir=args.output_dir,
        python_bin=args.python,
        launch_skip=args.launch_skip,
        launch_count=args.launch_count,
        use_sudo=args.sudo,
        ncu_bin=args.ncu,
        timeout=args.timeout,
        kernel_filter=args.kernel_filter,
        nvtx_include=args.nvtx_include,
        selection=args.selection,
        kernel_name_filter=args.kernel_name_filter,
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
