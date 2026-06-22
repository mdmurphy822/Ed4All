"""Side-by-side role-accuracy eval for a trained reasoner adapter.

Reads N rows from test.jsonl, runs greedy generation, and for each row writes
a self-contained markdown file under `eval/side_by_side/<run>/row_XX.md` with
the input chunk, the ground-truth labeling, and the predicted labeling laid
out for visual inspection.

Also writes `summary.md` and `summary.json` with parse rate, role accuracy,
and per-role F1.

Intentionally quiet on stdout — inspect the files in the output folder.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


def _parse_json_array(raw: str) -> list[dict] | None:
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _write_row_md(path: Path, i: int, system: str, user: str,
                  gold_by_id: dict[int, str], pred_by_id: dict[int, str],
                  raw_pred: str, parse_ok: bool) -> None:
    lines: list[str] = []
    lines.append(f"# row {i}")
    lines.append("")
    lines.append(f"- parse_ok: **{parse_ok}**")
    lines.append(f"- gold blocks: {len(gold_by_id)}  |  pred blocks: {len(pred_by_id)}")
    if gold_by_id:
        matches = sum(1 for bid, g in gold_by_id.items() if pred_by_id.get(bid) == g)
        acc = matches / len(gold_by_id)
        lines.append(f"- role accuracy on gold ids: {matches}/{len(gold_by_id)} = {acc:.2%}")
    lines.append("")

    lines.append("## side-by-side labels")
    lines.append("")
    lines.append("| id | gold | pred | match |")
    lines.append("|---:|:-----|:-----|:-----:|")
    for bid in sorted(set(gold_by_id) | set(pred_by_id)):
        g = gold_by_id.get(bid, "—")
        p = pred_by_id.get(bid, "—")
        mk = "✅" if g == p and g != "—" else ("·" if g == "—" or p == "—" else "❌")
        lines.append(f"| {bid} | {g} | {p} | {mk} |")
    lines.append("")

    lines.append("## raw predicted output")
    lines.append("")
    lines.append("```")
    lines.append(raw_pred.rstrip())
    lines.append("```")
    lines.append("")

    lines.append("## user prompt (input chunk)")
    lines.append("")
    lines.append("```")
    lines.append(user.rstrip())
    lines.append("```")
    lines.append("")

    lines.append("<details><summary>system prompt</summary>")
    lines.append("")
    lines.append("```")
    lines.append(system.rstrip())
    lines.append("```")
    lines.append("")
    lines.append("</details>")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--test-file", type=Path, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--run-name", default=None,
                    help="Folder under eval/side_by_side/. Defaults to "
                         "<adapter-stem>_<YYYYmmdd-HHMM>.")
    ap.add_argument("--out-root", type=Path, default=Path("eval/side_by_side"))
    args = ap.parse_args()

    run_name = args.run_name or (
        f"{Path(args.adapter).parent.name}_{datetime.now():%Y%m%d-%H%M}"
    )
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}")

    from dart_semantic.reason import load_reasoner, _generate

    print(f"[load] base={args.base_model} adapter={args.adapter}")
    reasoner = load_reasoner(base_model=args.base_model,
                             adapter_path=args.adapter)

    rows = []
    with open(args.test_file) as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            rows.append(json.loads(line))
    print(f"[data] {len(rows)} rows from {args.test_file}")

    parse_ok_n = 0
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    total_gt = 0
    total_match = 0

    for i, row in enumerate(rows):
        msgs = row["messages"]
        system = msgs[0]["content"]
        user = msgs[1]["content"]
        gold_raw = msgs[2]["content"]
        gold = _parse_json_array(gold_raw) or []
        gold_by_id = {int(g["id"]): g["role"] for g in gold
                      if isinstance(g, dict) and "id" in g and "role" in g}

        raw = _generate(reasoner, [msgs[0], msgs[1]], args.max_new_tokens)
        pred = _parse_json_array(raw)
        pred_by_id: dict[int, str] = {}
        parse_ok = pred is not None
        if parse_ok:
            parse_ok_n += 1
            for p in pred:
                if isinstance(p, dict) and "id" in p and "role" in p:
                    try:
                        pred_by_id[int(p["id"])] = p["role"]
                    except (TypeError, ValueError):
                        pass

        for bid in set(gold_by_id) | set(pred_by_id):
            g = gold_by_id.get(bid)
            p = pred_by_id.get(bid)
            if g is not None:
                total_gt += 1
                if g == p:
                    total_match += 1
                    tp[g] += 1
                else:
                    fn[g] += 1
                    if p is not None:
                        fp[p] += 1
            elif p is not None:
                fp[p] += 1

        _write_row_md(out_dir / f"row_{i:03d}.md", i, system, user,
                      gold_by_id, pred_by_id, raw, parse_ok)
        print(f"  row {i}: gold_n={len(gold_by_id)} pred_n={len(pred_by_id)} "
              f"parse_ok={parse_ok}")

    # ---- summary ----
    summary = {
        "adapter": str(args.adapter),
        "test_file": str(args.test_file),
        "n_rows": len(rows),
        "parse_ok": parse_ok_n,
        "parse_rate": parse_ok_n / len(rows) if rows else 0.0,
        "total_gold_blocks": total_gt,
        "total_correct": total_match,
        "role_accuracy": total_match / total_gt if total_gt else 0.0,
        "per_role": {},
    }
    roles = sorted(set(tp) | set(fp) | set(fn))
    for r in roles:
        t, f, n = tp[r], fp[r], fn[r]
        P = t / (t + f) if (t + f) else 0.0
        R = t / (t + n) if (t + n) else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) else 0.0
        summary["per_role"][r] = {"tp": t, "fp": f, "fn": n,
                                  "precision": P, "recall": R, "f1": F1}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        f"# reasoner eval — {run_name}",
        "",
        f"- adapter: `{args.adapter}`",
        f"- test: `{args.test_file}`",
        f"- rows: {len(rows)}",
        f"- parse rate: {parse_ok_n}/{len(rows)} = {summary['parse_rate']:.2%}",
        f"- role accuracy: {total_match}/{total_gt} = {summary['role_accuracy']:.2%}",
        "",
        "## per-role F1",
        "",
        "| role | tp | fp | fn | P | R | F1 |",
        "|:-----|---:|---:|---:|---:|---:|---:|",
    ]
    for r in roles:
        d = summary["per_role"][r]
        lines.append(f"| {r} | {d['tp']} | {d['fp']} | {d['fn']} | "
                     f"{d['precision']:.2%} | {d['recall']:.2%} | {d['f1']:.2%} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"[done] {out_dir}/summary.md")


if __name__ == "__main__":
    main()
