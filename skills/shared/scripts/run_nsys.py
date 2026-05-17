"""Execute NSYS profiling on a GPU kernel and parse results.

Runs Nsight Systems to collect timeline traces, parses the stats
output, and optionally exports to SQLite for fine-grained analysis.
Outputs structured JSON for Claude to interpret.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_nsys(nsys_bin: str | None = None) -> str:
    """Locate the nsys binary.

    Search order:
    1. Explicit path (--nsys argument)
    2. PATH lookup
    3. $CUDA_HOME/bin
    4. Common CUDA installation directories
    """
    if nsys_bin and nsys_bin != "nsys":
        if os.path.isfile(nsys_bin) and os.access(nsys_bin, os.X_OK):
            return nsys_bin
        found = shutil.which(nsys_bin)
        if found:
            return found
        raise RuntimeError(f"nsys not found at: {nsys_bin}")

    found = shutil.which("nsys")
    if found:
        return found

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidate = os.path.join(cuda_home, "bin", "nsys")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    for d in glob.glob("/usr/local/cuda*/bin"):
        candidate = os.path.join(d, "nsys")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    searched = ["PATH"]
    if cuda_home:
        searched.append(f"$CUDA_HOME={cuda_home}")
    searched.append("/usr/local/cuda*/bin (glob)")

    raise RuntimeError(
        "nsys not found. Searched: " + ", ".join(searched) + ".\n"
        "To locate nsys, try: find / -name nsys -type f 2>/dev/null\n"
        "Then pass the path: --nsys /path/to/nsys\n"
        "Or install Nsight Systems: https://developer.nvidia.com/nsight-systems"
    )


def run_nsys(
    script_path: str,
    output_dir: str,
    python_bin: str = "python",
    nsys_bin: str = "nsys",
    timeout: int = 120,
    trace_apis: str = "cuda,nvtx,osrt,cudnn,cublas",
    extra_args: list[str] | None = None,
) -> dict:
    """Run NSYS profiling and return parsed timeline summary."""
    nsys_bin = find_nsys(nsys_bin)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = str(output_dir / "nsys_report")

    cmd = [
        nsys_bin, "profile",
        "--stats=true",
        "--force-overwrite=true",
        f"--trace={trace_apis}",
        f"-o", report_path,
        python_bin, script_path,
    ]
    if extra_args:
        cmd = cmd[:7] + extra_args + cmd[7:]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(script_path).parent),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"NSYS timed out after {timeout}s",
            "command": " ".join(cmd),
        }

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    full_output = stdout + "\n" + stderr

    nsys_rep = report_path + ".nsys-rep"
    if not Path(nsys_rep).exists():
        nsys_rep_candidates = list(output_dir.glob("*.nsys-rep"))
        if nsys_rep_candidates:
            nsys_rep = str(nsys_rep_candidates[0])
        else:
            hints = []
            stderr_lower = stderr.lower()
            if "permission" in stderr_lower or "denied" in stderr_lower:
                hints.append(
                    "Permission denied. NSYS usually doesn't need sudo, "
                    "but some trace modes may require it. Try with sudo."
                )
            if "module" in stderr_lower and "not found" in stderr_lower:
                hints.append(
                    "Python module not found. Check Python env and "
                    "PYTHONPATH."
                )
            if not hints:
                hints.append(
                    "NSYS did not produce output. Check stderr below."
                )
            return {
                "success": False,
                "error": "NSYS did not produce .nsys-rep file",
                "hints": hints,
                "returncode": proc.returncode,
                "stderr": stderr[-3000:],
                "command": " ".join(cmd),
            }

    stats = parse_nsys_stats(full_output)

    sqlite_path = None
    sqlite_summary = None
    try:
        sqlite_path = export_to_sqlite(nsys_rep, output_dir, nsys_bin)
        if sqlite_path:
            sqlite_summary = analyze_sqlite(sqlite_path)
    except Exception as e:
        sqlite_summary = {"error": str(e)}

    return {
        "success": True,
        "nsys_rep_path": nsys_rep,
        "sqlite_path": sqlite_path,
        "stats": stats,
        "sqlite_summary": sqlite_summary,
        "command": " ".join(cmd),
    }


def parse_nsys_stats(output: str) -> dict:
    """Parse NSYS --stats=true text output into structured data."""
    result = {
        "cuda_kernels": [],
        "cuda_memcpy": [],
        "cuda_api": [],
        "os_runtime": [],
        "summary": {},
    }

    sections = _split_stats_sections(output)

    for section_name, section_text in sections.items():
        lower_name = section_name.lower()
        if "kern" in lower_name and "api" not in lower_name:
            result["cuda_kernels"] = _parse_stats_table(section_text)
        elif "memcpy" in lower_name or "memset" in lower_name:
            result["cuda_memcpy"] = _parse_stats_table(section_text)
        elif "cuda" in lower_name and "api" in lower_name:
            result["cuda_api"] = _parse_stats_table(section_text)
        elif "os runtime" in lower_name or "osrt" in lower_name:
            result["os_runtime"] = _parse_stats_table(section_text)

    total_kernel_time = sum(
        _safe_float(k.get("Total Time (ns)", k.get("Total", 0)))
        for k in result["cuda_kernels"]
    )
    total_memcpy_time = sum(
        _safe_float(m.get("Total Time (ns)", m.get("Total", 0)))
        for m in result["cuda_memcpy"]
    )
    result["summary"] = {
        "total_kernel_time_us": round(total_kernel_time / 1000, 2),
        "total_memcpy_time_us": round(total_memcpy_time / 1000, 2),
        "num_kernels": len(result["cuda_kernels"]),
        "num_memcpy_ops": len(result["cuda_memcpy"]),
        "num_api_calls": len(result["cuda_api"]),
    }

    if total_kernel_time + total_memcpy_time > 0:
        result["summary"]["kernel_time_pct"] = round(
            total_kernel_time / (total_kernel_time + total_memcpy_time) * 100, 1
        )
    return result


def _split_stats_sections(output: str) -> dict[str, str]:
    sections = {}
    header_pattern = re.compile(
        r"^\[.*?\]\s*(CUDA Kernel Statistics|CUDA Memory Operation Statistics|"
        r"CUDA API Statistics|OS Runtime API Statistics|"
        r".*?Kern.*?Stat.*|.*?Mem.*?Stat.*|.*?API.*?Stat.*)",
        re.MULTILINE | re.IGNORECASE,
    )

    stat_block_pattern = re.compile(
        r"((?:CUDA\s+)?(?:Kernel|Memory|Memcpy|Memset|API|OS\s*Runtime)"
        r"[^\n]*?(?:Statistics|Summary)[^\n]*)",
        re.IGNORECASE,
    )

    markers = list(stat_block_pattern.finditer(output))
    if not markers:
        markers = list(header_pattern.finditer(output))

    for i, match in enumerate(markers):
        name = match.group(0).strip()
        start = match.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(output)
        sections[name] = output[start:end]

    return sections


def _parse_stats_table(text: str) -> list[dict]:
    rows = []
    lines = text.strip().split("\n")
    header = None
    separator_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^[\-=\s]+$", stripped):
            separator_count += 1
            continue
        if header is None and separator_count >= 0:
            parts = re.split(r"\s{2,}", stripped)
            if len(parts) >= 3:
                header = parts
                continue
        if header is not None:
            parts = re.split(r"\s{2,}", stripped)
            if len(parts) >= len(header) - 1:
                row = {}
                for j, h in enumerate(header):
                    if j < len(parts):
                        row[h] = parts[j].strip()
                    else:
                        row[h] = ""
                rows.append(row)
            elif len(parts) >= 2:
                row = {}
                for j, h in enumerate(header):
                    if j < len(parts):
                        row[h] = parts[j].strip()
                rows.append(row)
    return rows


def _safe_float(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    v = str(v).replace(",", "").replace("%", "").strip()
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def export_to_sqlite(
    nsys_rep_path: str,
    output_dir: Path,
    nsys_bin: str = "nsys",
) -> str | None:
    """Export .nsys-rep to SQLite for detailed analysis."""
    sqlite_path = str(output_dir / "nsys_report.sqlite")
    cmd = [nsys_bin, "export", "-t", "sqlite", "-o", sqlite_path, nsys_rep_path]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        if Path(sqlite_path).exists():
            return sqlite_path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def analyze_sqlite(sqlite_path: str) -> dict:
    """Extract key timeline metrics from NSYS SQLite export."""
    import sqlite3
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    summary = {}

    try:
        cur = conn.execute("""
            SELECT
                demangledName as name,
                COUNT(*) as count,
                SUM(end - start) as total_ns,
                AVG(end - start) as avg_ns,
                MIN(end - start) as min_ns,
                MAX(end - start) as max_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            LEFT JOIN StringIds ON CUPTI_ACTIVITY_KIND_KERNEL.demangledName = StringIds.id
            GROUP BY demangledName
            ORDER BY total_ns DESC
            LIMIT 20
        """)
        summary["top_kernels"] = [dict(row) for row in cur.fetchall()]
    except Exception:
        try:
            cur = conn.execute("""
                SELECT
                    shortName as name,
                    COUNT(*) as count,
                    SUM(end - start) as total_ns,
                    AVG(end - start) as avg_ns,
                    MIN(end - start) as min_ns,
                    MAX(end - start) as max_ns
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                GROUP BY shortName
                ORDER BY total_ns DESC
                LIMIT 20
            """)
            summary["top_kernels"] = [dict(row) for row in cur.fetchall()]
        except Exception as e:
            summary["top_kernels_error"] = str(e)

    try:
        cur = conn.execute("""
            SELECT
                copyKind,
                COUNT(*) as count,
                SUM(end - start) as total_ns,
                SUM(bytes) as total_bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            GROUP BY copyKind
        """)
        memcpy_kinds = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "HtoD_Pinned"}
        memcpy_rows = []
        for row in cur.fetchall():
            r = dict(row)
            r["direction"] = memcpy_kinds.get(r.get("copyKind", 0), f"kind_{r.get('copyKind')}")
            memcpy_rows.append(r)
        summary["memcpy"] = memcpy_rows
    except Exception as e:
        summary["memcpy_error"] = str(e)

    try:
        cur = conn.execute("""
            SELECT
                MIN(start) as first_event,
                MAX(end) as last_event
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        """)
        row = cur.fetchone()
        if row and row["first_event"] is not None:
            wall_time_ns = row["last_event"] - row["first_event"]
            summary["wall_time_us"] = round(wall_time_ns / 1000, 2)

            cur2 = conn.execute(
                "SELECT SUM(end - start) as busy FROM CUPTI_ACTIVITY_KIND_KERNEL"
            )
            busy = cur2.fetchone()
            if busy and busy["busy"]:
                summary["gpu_utilization_pct"] = round(
                    busy["busy"] / wall_time_ns * 100, 1
                )
    except Exception as e:
        summary["utilization_error"] = str(e)

    try:
        cur = conn.execute("""
            SELECT
                nameId,
                COUNT(*) as count,
                SUM(end - start) as total_ns,
                AVG(end - start) as avg_ns
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            GROUP BY nameId
            ORDER BY total_ns DESC
            LIMIT 15
        """)
        summary["cuda_api_calls"] = [dict(row) for row in cur.fetchall()]
    except Exception:
        pass

    conn.close()
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run NSYS profiling on a GPU kernel"
    )
    parser.add_argument("script", help="Python script to profile")
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory (default: /tmp/gpu_profile_nsys)",
    )
    parser.add_argument("--python", default="python", help="Python binary")
    parser.add_argument("--nsys", default="nsys", help="NSYS binary path")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--trace", default="cuda,nvtx,osrt,cudnn,cublas",
        help="NSYS trace APIs",
    )
    parser.add_argument("--no-sqlite", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = tempfile.mkdtemp(prefix="gpu_profile_nsys_")

    result = run_nsys(
        script_path=args.script,
        output_dir=args.output_dir,
        python_bin=args.python,
        nsys_bin=args.nsys,
        timeout=args.timeout,
        trace_apis=args.trace,
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
