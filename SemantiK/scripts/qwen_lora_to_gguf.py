"""Phase 1.5 — convert a trained Qwen LoRA adapter to a quantized GGUF.

Pipeline:

    1. Merge the LoRA into the base Qwen3-4B in fp16. Output:
       ``<work>/merged_fp16/`` (HF-format directory).
    2. Convert merged HF dir to fp16 GGUF via
       ``$LLAMA_CPP_REPO/convert_hf_to_gguf.py``.
    3. Quantize fp16 GGUF to ``--quant`` (default Q4_K_M) via
       ``$LLAMA_CPP_REPO/build/bin/llama-quantize``.
    4. Atomic rename the quantized blob to ``--out``. The atomic step
       guards against a half-written GGUF being picked up by a runtime
       reload.

Optional:
    --keep-merged   keep the fp16 HF directory (useful for debugging).
    --overwrite     allow ``--out`` to already exist.

Sidecar metadata:
    The script writes ``<out>.metadata.json`` next to the final GGUF
    with adapter_version + base model + quant + sha256(blob). The
    runtime (``LlamaCppRuntime.load``) reads this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# llama.cpp checkout location is machine-specific; supply it via the
# LLAMA_CPP_REPO env var (no hardcoded absolute path in tracked code). Falls
# back to ``~/llama.cpp`` for the common single-user layout.
LLAMA_CPP_REPO = Path(
    os.environ.get("LLAMA_CPP_REPO", str(Path.home() / "llama.cpp"))
).expanduser()
CONVERT_SCRIPT = LLAMA_CPP_REPO / "convert_hf_to_gguf.py"
QUANTIZE_BIN = LLAMA_CPP_REPO / "build" / "bin" / "llama-quantize"


def _check_tools() -> None:
    if not CONVERT_SCRIPT.exists():
        raise FileNotFoundError(
            f"llama.cpp convert script missing: {CONVERT_SCRIPT}. "
            f"Clone https://github.com/ggerganov/llama.cpp."
        )
    if not QUANTIZE_BIN.exists():
        raise FileNotFoundError(
            f"llama-quantize missing: {QUANTIZE_BIN}. "
            f"Build llama.cpp: `cmake -B build -DGGML_CUDA=ON && cmake --build build -j`."
        )


def _adapter_metadata(adapter_dir: Path) -> dict:
    meta_path = adapter_dir / "adapter_metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    # If the adapter was saved without our metadata writer (e.g.
    # legacy reasoner), invent a minimal stub so downstream runtime
    # has a non-null version string.
    return {
        "adapter_version": f"unversioned-{int(time.time())}",
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _merge_lora(adapter_dir: Path, base_model: str, work: Path) -> Path:
    """Merge LoRA into base; return path to the merged HF directory."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    merged_dir = work / "merged_fp16"
    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] loading base {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    print(f"[merge] applying LoRA from {adapter_dir}")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    print("[merge] merge_and_unload()")
    model = model.merge_and_unload()
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.save_pretrained(str(merged_dir))
    return merged_dir


def _convert_to_gguf(merged_dir: Path, out_fp16: Path) -> None:
    cmd = [
        sys.executable, str(CONVERT_SCRIPT),
        str(merged_dir),
        "--outfile", str(out_fp16),
        "--outtype", "f16",
    ]
    print(f"[convert] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _quantize(in_fp16: Path, out_quant: Path, quant: str) -> None:
    cmd = [str(QUANTIZE_BIN), str(in_fp16), str(out_quant), quant]
    print(f"[quantize] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter-dir", required=True, type=Path,
                    help="Path to the trained LoRA dir (the one with adapter_config.json).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Final quantized GGUF path.")
    ap.add_argument("--quant", default="Q4_K_M",
                    help="llama-quantize type. Q4_K_M is the 3060-friendly default.")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--keep-merged", action="store_true",
                    help="Don't delete the intermediate fp16 HF dir.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow --out to already exist.")
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="Override scratch dir (default: tempfile).")
    args = ap.parse_args()

    if not args.adapter_dir.exists():
        print(f"adapter dir missing: {args.adapter_dir}", file=sys.stderr)
        return 2
    if args.out.exists() and not args.overwrite:
        print(f"--out exists: {args.out}; pass --overwrite", file=sys.stderr)
        return 3

    try:
        _check_tools()
    except FileNotFoundError as exc:
        print(f"toolchain error: {exc}", file=sys.stderr)
        return 4

    work = args.work_dir or Path(tempfile.mkdtemp(prefix="qwen_gguf_"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"[work] {work}")

    try:
        merged_dir = _merge_lora(args.adapter_dir, args.base_model, work)
        fp16_path = work / "model.f16.gguf"
        _convert_to_gguf(merged_dir, fp16_path)
        quantized_tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        _quantize(fp16_path, quantized_tmp, args.quant)
        # Atomic rename — guarantees the runtime never sees a partial blob.
        os.replace(quantized_tmp, args.out)
        print(f"[done] {args.out}")

        # Sidecar metadata.
        adapter_meta = _adapter_metadata(args.adapter_dir)
        sidecar = {
            "adapter_version": adapter_meta.get("adapter_version", "unknown"),
            "base_model": args.base_model,
            "quant": args.quant,
            "sha256": _sha256(args.out),
            "adapter_metadata": adapter_meta,
            "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        }
        sidecar_path = args.out.with_suffix(args.out.suffix + ".metadata.json")
        sidecar_path.write_text(json.dumps(sidecar, indent=2))
        print(f"[meta] {sidecar_path}")
    finally:
        if not args.keep_merged and args.work_dir is None:
            shutil.rmtree(work, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
