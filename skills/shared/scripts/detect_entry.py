from __future__ import annotations

"""Detect the executable entry point of a GPU kernel Python file.

Analyzes a .py file to determine how it can be invoked for profiling:
- Has __main__ block: directly runnable
- Has benchmark/test/main functions: generate a wrapper call
- Only kernel definitions: report what's available for the user to decide

Supports: Triton, Numba CUDA, TileLang, PyCUDA, CuPy, PyTorch JIT (load_inline).
"""

import argparse
import ast
import json
import sys
from pathlib import Path

FRAMEWORK_REGISTRY = {
    "triton": {
        "imports": ["triton"],
        "kernel_decorators": ["triton.jit", "triton.autotune"],
        "label": "Triton",
    },
    "numba_cuda": {
        "imports": ["numba.cuda", "numba"],
        "kernel_decorators": ["cuda.jit"],
        "label": "Numba CUDA",
    },
    "tilelang": {
        "imports": ["tilelang"],
        "kernel_decorators": ["tilelang.jit", "tilelang.autotune", "T.prim_func"],
        "label": "TileLang",
    },
    "pycuda": {
        "imports": ["pycuda"],
        "kernel_decorators": [],
        "label": "PyCUDA",
    },
    "cupy": {
        "imports": ["cupy"],
        "kernel_decorators": [],
        "label": "CuPy",
    },
    "native": {
        "imports": ["ctypes", "cffi"],
        "kernel_decorators": [],
        "label": "Native .so / ctypes",
    },
    "torch_jit": {
        "imports": ["torch.utils.cpp_extension"],
        "kernel_decorators": [],
        "label": "PyTorch JIT (load_inline)",
    },
    "torch_cuda": {
        "imports": ["torch.cuda", "torch"],
        "kernel_decorators": [],
        "label": "PyTorch CUDA",
    },
    "triton_aot": {
        "imports": [],
        "kernel_decorators": [],
        "label": "Triton AOT (triton.compile)",
    },
    "jax": {
        "imports": ["jax"],
        "kernel_decorators": [],
        "label": "JAX",
    },
    "tensorrt": {
        "imports": ["tensorrt"],
        "kernel_decorators": [],
        "label": "TensorRT",
    },
    "tensorflow": {
        "imports": ["tensorflow"],
        "kernel_decorators": [],
        "label": "TensorFlow",
    },
}


def analyze_kernel_file(filepath: str) -> dict:
    """Analyze a Python file to detect GPU kernel entry points.

    Args:
        filepath: Path to the .py file to analyze.

    Returns:
        Dict with framework, kernels, entry_type, runnable, description, etc.
        On error, returns dict with "error" key.
    """
    fpath = Path(filepath)
    if not fpath.exists():
        return {"error": f"File not found: {filepath}", "filepath": str(fpath)}
    if not fpath.is_file():
        return {"error": f"Not a file: {filepath}", "filepath": str(fpath)}
    if not fpath.suffix == ".py":
        return {"error": f"Not a Python file: {filepath}", "filepath": str(fpath)}

    try:
        source = fpath.read_text()
    except PermissionError:
        return {
            "error": f"Permission denied reading: {filepath}",
            "filepath": str(fpath.resolve()),
        }

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        return {
            "error": f"Syntax error in {filepath}: {e.msg} (line {e.lineno})",
            "filepath": str(fpath.resolve()),
        }

    detected_frameworks = _detect_frameworks(tree, source)
    primary = detected_frameworks[0] if detected_frameworks else "unknown"

    kernel_decorators_all = []
    for fw in detected_frameworks:
        kernel_decorators_all.extend(FRAMEWORK_REGISTRY[fw]["kernel_decorators"])

    kernel_names = set()
    all_functions = {}

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        decorators = _extract_decorators(node)
        is_kernel = any(
            any(kd in d for kd in kernel_decorators_all) for d in decorators
        )

        func_info = {
            "name": node.name,
            "args": _extract_args(node),
            "decorators": decorators,
            "is_kernel": is_kernel,
            "lineno": node.lineno,
        }
        all_functions[node.name] = func_info

        if is_kernel:
            kernel_names.add(node.name)

    kernels = [f for f in all_functions.values() if f["is_kernel"]]
    launch_functions = []
    for name, func_info in all_functions.items():
        if name in kernel_names:
            continue
        if _calls_kernel(tree, name, kernel_names):
            launch_functions.append(func_info)

    has_main_block = _has_main_block(tree)

    result = {
        "filepath": str(fpath.resolve()),
        "framework": primary,
        "frameworks_detected": detected_frameworks,
        "entry_type": "kernel_only",
        "runnable": False,
        "main_block": has_main_block,
        "kernels": kernels,
        "launch_functions": launch_functions,
        "imports": {
            fw: fw in detected_frameworks for fw in FRAMEWORK_REGISTRY
        },
        "description": "",
    }
    for fw in detected_frameworks:
        if fw == "torch_cuda":
            result["imports"]["torch_cuda"] = True

    if has_main_block:
        result["entry_type"] = "main_block"
        result["runnable"] = True
        result["description"] = (
            f"File has __main__ block, directly runnable with `python {filepath}`"
        )
    elif launch_functions:
        result["entry_type"] = "callable"
        result["runnable"] = False
        names = [f["name"] for f in launch_functions]
        result["description"] = (
            f"Found launch functions: {names}. Need a wrapper to call them."
        )
    elif kernels:
        result["entry_type"] = "kernel_only"
        result["runnable"] = False
        names = [f["name"] for f in kernels]
        result["description"] = (
            f"Found {primary} kernels: {names}. "
            "No launch function or __main__ block. Need user to specify how to invoke."
        )
    else:
        result["entry_type"] = "kernel_only"
        result["runnable"] = False
        result["description"] = (
            "No GPU kernel decorators or entry points detected. "
            "If this is a PyTorch JIT kernel (load_inline / cpp_extension) or raw "
            "CUDA C++ compiled at runtime, NCU/NSYS will still capture all CUDA "
            "kernel launches. Proceed with profiling — kernel name filters may "
            "need to match mangled names with template parameters."
        )

    return result


def _detect_frameworks(tree: ast.Module, source: str = "") -> list[str]:
    """Detect GPU frameworks by scanning imports and source text."""
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module)

    detected = []
    for fw_key, fw_info in FRAMEWORK_REGISTRY.items():
        for imp in fw_info["imports"]:
            if imp in imported_modules or any(
                m.startswith(imp + ".") for m in imported_modules
            ):
                detected.append(fw_key)
                break

    if not detected and source:
        if any(kw in source for kw in ["cuda.jit", "@cuda", "pycuda"]):
            if "numba" in source or "cuda.jit" in source:
                detected.append("numba_cuda")
            elif "pycuda" in source:
                detected.append("pycuda")
        if any(kw in source for kw in ["load_inline", "cpp_extension", "load_library",
                                         "CUDAExtension"]):
            detected.append("torch_jit")
        if any(kw in source for kw in ["RawKernel", "RawModule"]):
            detected.append("cupy")
        if any(kw in source for kw in ["T.prim_func", "T.macro"]):
            detected.append("tilelang")
        if any(kw in source for kw in ["ctypes.CDLL", "cdll.LoadLibrary",
                                         "ffi.dlopen"]):
            detected.append("native")
        if any(kw in source for kw in ["triton.compile"]):
            detected.append("triton_aot")
        if any(kw in source for kw in ["jax.custom_call", "register_custom_call_target"]):
            detected.append("jax")
        if any(kw in source for kw in ["tensorrt.Builder"]):
            detected.append("tensorrt")
        if any(kw in source for kw in ["tf.load_op_library"]):
            detected.append("tensorflow")

    return detected if detected else ["torch_cuda"]


def _is_constant_node(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) or (
        hasattr(ast, 'Str') and isinstance(node, ast.Str)
    )


def _is_main_constant(node: ast.AST) -> bool:
    """Check if an AST node is the string literal '__main__' (py37+ compat)."""
    if _is_constant_node(node):
        val = node.value if hasattr(node, 'value') else node.s
        return val == "__main__"
    return False


def _has_main_block(tree: ast.Module) -> bool:
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.If):
            try:
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                ):
                    for comp in test.comparators:
                        if _is_main_constant(comp):
                            return True
            except (AttributeError, IndexError):
                pass
    return False


def _extract_decorators(node: ast.FunctionDef) -> list[str]:
    decorators = []
    for dec in node.decorator_list:
        try:
            decorators.append(_ast_to_source(dec))
        except Exception:
            decorators.append("<unknown_decorator>")
    return decorators


def _ast_to_source(node: ast.AST) -> str:
    """Python 3.7 compatible AST to source conversion (no ast.unparse)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _ast_to_source(node.value) + "." + node.attr
    if isinstance(node, ast.Call):
        args_str = ", ".join(_ast_to_source(a) for a in node.args)
        for kw in node.keywords:
            args_str += ", %s=%s" % (kw.arg, _ast_to_source(kw.value))
        return _ast_to_source(node.func) + "(" + args_str + ")"
    if _is_constant_node(node):
        val = node.value if hasattr(node, 'value') else node.s
        return repr(val) if isinstance(val, str) else str(val)
    if hasattr(ast, 'Num') and isinstance(node, ast.Num):
        return str(node.n)
    if isinstance(node, ast.Subscript):
        return (_ast_to_source(node.value) + "["
                + _ast_to_source(node.slice) + "]")
    if hasattr(ast, 'Index') and isinstance(node, ast.Index):
        return _ast_to_source(node.value)
    if isinstance(node, ast.Tuple):
        return ", ".join(_ast_to_source(e) for e in node.elts)
    return "<unknown>"


def _extract_args(node: ast.FunctionDef) -> list[str]:
    return [arg.arg for arg in node.args.args]


def _calls_kernel(
    tree: ast.Module, func_name: str, kernel_names: set[str]
) -> bool:
    """Check if a function calls any of the detected kernel functions.

    Handles Triton grid launch syntax kernel[grid](args),
    Numba CUDA kernel[grid, block](args), TileLang JITKernel calls,
    and PyCUDA prepared_call patterns.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != func_name:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            callee = child.func
            if isinstance(callee, ast.Name) and callee.id in kernel_names:
                return True
            if isinstance(callee, ast.Subscript):
                if (
                    isinstance(callee.value, ast.Name)
                    and callee.value.id in kernel_names
                ):
                    return True
            if isinstance(callee, ast.Attribute):
                if _attr_chain_contains(callee, kernel_names):
                    return True
    return False


def _attr_chain_contains(node: ast.Attribute, names: set[str]) -> bool:
    """Check if any name in the attribute chain (e.g. obj.attr.func) matches."""
    current = node
    while isinstance(current, ast.Attribute):
        if current.attr in names:
            return True
        current = current.value
    if isinstance(current, ast.Name) and current.id in names:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Detect entry point of a GPU kernel file"
    )
    parser.add_argument("filepath", help="Path to the .py file to analyze")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = analyze_kernel_file(args.filepath)

    if args.json:
        print(json.dumps(result, indent=2))
        if "error" in result:
            sys.exit(1)
    else:
        if "error" in result:
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)
        print(f"File: {result['filepath']}")
        print(f"Framework: {result['framework']}")
        print(f"Entry type: {result['entry_type']}")
        print(f"Runnable: {result['runnable']}")
        print(f"Description: {result['description']}")
        if result["kernels"]:
            print(f"Kernels: {[k['name'] for k in result['kernels']]}")
        if result["launch_functions"]:
            print(
                "Launch functions: "
                f"{[f['name'] for f in result['launch_functions']]}"
            )


if __name__ == "__main__":
    main()
